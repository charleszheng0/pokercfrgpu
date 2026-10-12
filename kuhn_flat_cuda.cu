#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>

#define CHECK(x) do { \
    cudaError_t e = (x); \
    if (e != cudaSuccess) { \
        fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(e)); \
        exit(1); \
    } \
} while (0)

// 4 nonterminal histories: "", "c", "b", "cb"
// 6 possible ordered card deals
// 12 information sets
constexpr int H = 4;
constexpr int D = 6;
constexpr int N = H * D;
constexpr int I = 12;

// Cards: J=0, Q=1, K=2
// All possible deals (P1 card, P2 card)
__constant__ int deals[12] = {
    0,1, 0,2, 1,0, 1,2, 2,0, 2,1
};

// Parent history for each nonterminal history
__constant__ int parentHist[4] = {-1, 0, 0, 1};

// Map a game state to its information set
__device__ __forceinline__
int info(int deal, int hist) {
    int player = (hist == 0 || hist == 3) ? 0 : 1;
    return hist * 3 + deals[deal * 2 + player];
}

// Terminal payoff from P1's perspective
__device__ __forceinline__
float payoff(int deal, int hist, int action) {
    int p1 = deals[deal * 2];
    int p2 = deals[deal * 2 + 1];

    // P2 folds after P1 bets
    if (hist == 2 && action == 0)
        return 1.f;

    // P1 folds after P2 bets
    if (hist == 3 && action == 0)
        return -1.f;

    // Showdown
    float winner = (p1 > p2) ? 1.f : -1.f;

    // Check-check = +/-1
    // Bet-call = +/-2
    return winner * (hist == 1 ? 1.f : 2.f);
}

// ========================================
// KERNEL 1: REGRET MATCHING
// ========================================

__global__
void match(const float* regrets, float* strat) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= I) return;

    float a = fmaxf(regrets[2 * i], 0.f);
    float b = fmaxf(regrets[2 * i + 1], 0.f);

    float sum = a + b;

    strat[2 * i] =
        (sum > 0.f) ? a / sum : 0.5f;

    strat[2 * i + 1] =
        (sum > 0.f) ? b / sum : 0.5f;
}

// ========================================
// KERNEL 2: REACH PROBABILITIES
// ========================================

__global__
void reachDepth(
    int hist,
    const float* strat,
    float* r1,
    float* r2
) {
    int d = blockIdx.x * blockDim.x + threadIdx.x;

    if (d >= D) return;

    int n = d * H + hist;

    if (hist == 0) {
        r1[n] = 1.f;
        r2[n] = 1.f;
        return;
    }

    int parent = parentHist[hist];
    int pn = d * H + parent;

    int ii = info(d, parent);

    // Action taken to reach this history
    int action = (hist == 1) ? 0 : 1;

    float p = strat[2 * ii + action];

    bool p1turn = (parent == 0 || parent == 3);

    r1[n] = r1[pn] * (p1turn ? p : 1.f);
    r2[n] = r2[pn] * (p1turn ? 1.f : p);
}

// ========================================
// KERNEL 3: BOTTOM-UP TREE EVALUATION
// ========================================

__global__
void evaluate(
    int hist,
    const float* strat,
    float* val,
    float* actionVals
) {
    int d = blockIdx.x * blockDim.x + threadIdx.x;

    if (d >= D) return;

    int n = d * H + hist;
    int ii = info(d, hist);

    float v0, v1;

    if (hist == 3 || hist == 2) {
        v0 = payoff(d, hist, 0);
        v1 = payoff(d, hist, 1);
    }
    else if (hist == 1) {
        v0 = payoff(d, 1, 0);
        v1 = val[d * H + 3];
    }
    else {
        v0 = val[d * H + 1];
        v1 = val[d * H + 2];
    }

    actionVals[2 * n] = v0;
    actionVals[2 * n + 1] = v1;

    val[n] =
        strat[2 * ii] * v0 +
        strat[2 * ii + 1] * v1;
}

// ========================================
// KERNEL 4: CFR REGRET UPDATES
// ========================================

__global__
void update(
    const float* strat,
    const float* r1,
    const float* r2,
    const float* val,
    const float* av,
    float* regrets,
    float* sums
) {
    int ii = blockIdx.x * blockDim.x + threadIdx.x;

    if (ii >= I) return;

    int hist = ii / 3;
    bool p1turn = (hist == 0 || hist == 3);

    float delta0 = 0.f;
    float delta1 = 0.f;
    float weight = 0.f;

    // Aggregate all card deals belonging to this infoset
    for (int d = 0; d < D; d++) {
        if (info(d, hist) != ii)
            continue;

        int n = d * H + hist;

        float opponentReach = p1turn ? r2[n] : r1[n];
        float ownReach = p1turn ? r1[n] : r2[n];

        float sign = p1turn ? 1.f : -1.f;

        delta0 += opponentReach * sign *
                  (av[2 * n] - val[n]);

        delta1 += opponentReach * sign *
                  (av[2 * n + 1] - val[n]);

        weight += ownReach;
    }

    regrets[2 * ii] += delta0;
    regrets[2 * ii + 1] += delta1;

    sums[2 * ii] += weight * strat[2 * ii];
    sums[2 * ii + 1] += weight * strat[2 * ii + 1];
}

// ========================================
// MAIN
// ========================================

int main(int argc, char** argv) {

    int iterations = argc > 1 ? atoi(argv[1]) : 12342;

    float *regrets, *strat;
    float *r1, *r2;
    float *val, *actionVals;
    float *strategySum;

    // Allocate GPU memory
    CHECK(cudaMalloc(&regrets, I * 2 * sizeof(float)));
    CHECK(cudaMalloc(&strat, I * 2 * sizeof(float)));

    CHECK(cudaMalloc(&r1, N * sizeof(float)));
    CHECK(cudaMalloc(&r2, N * sizeof(float)));

    CHECK(cudaMalloc(&val, N * sizeof(float)));
    CHECK(cudaMalloc(&actionVals, N * 2 * sizeof(float)));

    CHECK(cudaMalloc(&strategySum, I * 2 * sizeof(float)));

    // Initialize
    CHECK(cudaMemset(regrets, 0, I * 2 * sizeof(float)));
    CHECK(cudaMemset(strategySum, 0, I * 2 * sizeof(float)));

    cudaEvent_t start, stop;
    CHECK(cudaEventCreate(&start));
    CHECK(cudaEventCreate(&stop));

    CHECK(cudaEventRecord(start));

    // ========================================
    // CFR TRAINING LOOP
    // ========================================

    for (int t = 0; t < iterations; t++) {

        // 1. Regret matching
        match<<<1, 32>>>(regrets, strat);

        // 2. Calculate reach probabilities top-down
        for (int h = 0; h < H; h++) {
            reachDepth<<<1, 32>>>(h, strat, r1, r2);
        }

        // 3. Evaluate game tree bottom-up
        for (int h = H - 1; h >= 0; h--) {
            evaluate<<<1, 32>>>(
                h, strat, val, actionVals
            );
        }

        // 4. Update regrets and average strategies
        update<<<1, 32>>>(
            strat, r1, r2, val,
            actionVals, regrets, strategySum
        );
    }

    CHECK(cudaGetLastError());

    CHECK(cudaEventRecord(stop));
    CHECK(cudaEventSynchronize(stop));

    float ms;
    CHECK(cudaEventElapsedTime(&ms, start, stop));

    // Copy average strategy to CPU
    float host[I * 2];

    CHECK(cudaMemcpy(
        host,
        strategySum,
        sizeof(host),
        cudaMemcpyDeviceToHost
    ));

    const char* histories[4] = {"", "c", "b", "cb"};
    const char* cards[3] = {"J", "Q", "K"};

    printf("Iterations: %d\n", iterations);
    printf("GPU elapsed: %.2f ms\n\n", ms);

    // Print learned average strategies
    for (int h = 0; h < H; h++) {
        for (int c = 0; c < 3; c++) {

            int i = h * 3 + c;

            float total =
                host[2 * i] + host[2 * i + 1];

            float p0 = total > 0
                ? host[2 * i] / total : 0.5f;

            float p1 = total > 0
                ? host[2 * i + 1] / total : 0.5f;

            printf(
                "%s%s: %.4f %.4f\n",
                cards[c], histories[h], p0, p1
            );
        }
    }

    // Cleanup
    CHECK(cudaFree(regrets));
    CHECK(cudaFree(strat));
    CHECK(cudaFree(r1));
    CHECK(cudaFree(r2));
    CHECK(cudaFree(val));
    CHECK(cudaFree(actionVals));
    CHECK(cudaFree(strategySum));

    CHECK(cudaEventDestroy(start));
    CHECK(cudaEventDestroy(stop));

    return 0;
}