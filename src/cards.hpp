// Cards and the hand evaluator.
// A card is rank * 4 + suit. Rank 0 = deuce ... 12 = ace. Suit 0..3 = c, d, h, s.
#pragma once
#include <cstdint>
#include <string>
#include <vector>

namespace poker {

using Card = uint8_t;
constexpr int kNumCards = 52;

inline int rank_of(Card c) { return c >> 2; }
inline int suit_of(Card c) { return c & 3; }
inline Card make_card(int rank, int suit) { return static_cast<Card>(rank * 4 + suit); }

int parse_rank(char ch);                       // -1 if not a rank
int parse_card(const std::string& s);          // -1 if not a card, e.g. "Ah"
std::string card_str(Card c);
std::vector<Card> parse_cards(const std::string& s);  // throws std::invalid_argument
std::string cards_str(const std::vector<Card>& cards);

// Strength of the best 5-card hand in 5..7 cards. Higher value = stronger hand.
// Bits 20..23 hold the category (0 high card .. 8 straight flush).
uint32_t eval_hand(const Card* cards, int n);
inline int hand_category(uint32_t strength) { return static_cast<int>(strength >> 20); }
const char* category_name(int category);

// Key that is the same for every board you get by renaming suits (and for any card order),
// e.g. "Ks 9d 5c" and "Kh 9s 5d". Use it only when both ranges are suit-symmetric.
std::string canonical_board_key(const std::vector<Card>& cards);

}  // namespace poker
