// Starting spots for the training database: 100 bb 6-max cash, no rake.
// These are hand-built "house ranges" (see docs/research.md). They are approximations of
// real preflop strategy, not solver output. Solutions are exact for these ranges.
#pragma once
#include <string>
#include <vector>

namespace poker {

struct Preset {
  std::string name;
  std::string oop_pos, ip_pos;  // position names; OOP acts first after the flop
  std::string oop, ip;          // ranges at the start of the flop
  double pot, stack;            // big blinds at the flop: pot, effective stack behind
  std::string preflop;          // one line that tells the preflop action
};

inline const std::vector<Preset>& presets() {
  static const std::string kBtnOpen = "22+,A2s+,K4s+,Q6s+,J7s+,T7s+,97s+,86s+,75s+,64s+,54s,A2o+,K9o+,Q9o+,J9o+,T9o";
  static const std::string kBtnCall3b =
      "QQ:0.5,JJ-77,66-22:0.5,AKs:0.3,AQs,AJs,ATs,A9s-A6s:0.5,KQs,KJs,KTs,K9s:0.5,QJs,QTs,JTs,J9s:0.5,T9s,98s,87s,"
      "76s,65s,54s:0.5,AKo:0.3,AQo,AJo:0.5,KQo:0.5";
  static const std::vector<Preset> p = {
      {"BTN vs BB SRP", "BB", "BTN",
       "22-TT,JJ:0.5,A2s-AQs,K2s+,Q4s+,J6s+,T6s+,96s+,85s+,75s+,64s+,54s,43s,A2o-AJo,K8o+,Q9o+,J9o+,T9o,98o,87o",
       kBtnOpen, 5.5, 97.5, "BTN raises to 2.5 bb, SB folds, BB calls"},
      {"CO vs BB SRP", "BB", "CO",
       "22-TT,JJ:0.5,A2s-AQs,K4s+,Q6s+,J7s+,T7s+,97s+,86s+,75s+,65s,54s,A8o-AJo,KTo+,QTo+,JTo,T9o",
       "22+,A2s+,K6s+,Q8s+,J8s+,T8s+,97s+,87s,76s,65s,54s,A9o+,KTo+,QTo+,JTo", 5.5, 97.5,
       "CO raises to 2.5 bb, BTN and SB fold, BB calls"},
      {"UTG vs BB SRP", "BB", "UTG",
       "22-TT,JJ:0.5,A2s-AJs,AQs:0.5,K9s+,Q9s+,J9s+,T8s+,97s+,87s,76s,65s,54s,ATo-AJo,KQo,KJo,QJo",
       "22+,A2s+,KTs+,QTs+,JTs,T9s,98s,87s,76s,65s,54s,AJo+,KQo", 5.5, 97.5,
       "UTG raises to 2.5 bb, all fold to BB, BB calls"},
      {"SB vs BB SRP", "SB", "BB", "22+,A2s+,K2s+,Q5s+,J7s+,T7s+,96s+,86s+,75s+,65s,54s,A2o+,K8o+,Q9o+,J9o+,T9o",
       "22-99,TT:0.5,A2s-AJs,K2s+,Q2s+,J4s+,T6s+,95s+,85s+,74s+,64s+,53s+,43s,A2o-AJo,K5o+,Q8o+,J8o+,T8o+,98o,87o,76o",
       6, 97, "All fold to SB, SB raises to 3 bb, BB calls"},
      {"BB 3-bet vs BTN", "BB", "BTN",
       "QQ+,JJ:0.5,AKs,AKo,AQs,AQo:0.5,AJs:0.5,A5s-A2s,KQs:0.5,K9s:0.5,98s:0.5,87s:0.5,76s:0.5,65s:0.5", kBtnCall3b,
       22.5, 89, "BTN raises to 2.5 bb, SB folds, BB 3-bets to 11 bb, BTN calls"},
      {"SB 3-bet vs BTN", "SB", "BTN",
       "JJ+,TT:0.5,AKs,AKo,AQs,AQo:0.5,AJs:0.5,A5s-A2s,KQs,K9s:0.5,QJs:0.5,T9s:0.5,98s:0.5,87s:0.5,76s:0.5", kBtnCall3b,
       21, 90, "BTN raises to 2.5 bb, SB 3-bets to 10 bb, BB folds, BTN calls"},
  };
  return p;
}

}  // namespace poker
