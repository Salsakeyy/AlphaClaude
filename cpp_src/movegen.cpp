#include "movegen.h"
#include "bitboard.h"

namespace alphaclaude {

// Generate pawn moves for given color
template<Color Us>
static void generate_pawn_moves(const Position& pos, MoveList& list, Bitboard target) {
    constexpr Color Them = (Us == WHITE) ? BLACK : WHITE;
    constexpr Direction Up = (Us == WHITE) ? NORTH : SOUTH;
    constexpr Direction UpLeft = (Us == WHITE) ? NORTH_WEST : SOUTH_WEST;
    constexpr Direction UpRight = (Us == WHITE) ? NORTH_EAST : SOUTH_EAST;
    constexpr Bitboard Rank3 = (Us == WHITE) ? Rank3BB : Rank6BB;
    constexpr Bitboard Rank7 = (Us == WHITE) ? Rank7BB : Rank2BB;

    Bitboard pawns = pos.pieces(Us, PAWN);
    Bitboard enemies = pos.pieces(Them);
    Bitboard empty = ~pos.pieces();

    Bitboard promoting = pawns & Rank7;
    Bitboard not_promoting = pawns & ~Rank7;

    // Single push (non-promoting)
    Bitboard single_push = shift<Up>(not_promoting) & empty;
    Bitboard double_push = shift<Up>(single_push & Rank3) & empty;

    single_push &= target;
    double_push &= target;

    while (single_push) {
        Square to = pop_lsb(single_push);
        Square from = to - Up;
        list.add(Move::make(from, to));
    }

    while (double_push) {
        Square to = pop_lsb(double_push);
        Square from = to - Up - Up;
        list.add(Move::make(from, to));
    }

    // Captures (non-promoting)
    Bitboard cap_left = shift<UpLeft>(not_promoting) & enemies & target;
    Bitboard cap_right = shift<UpRight>(not_promoting) & enemies & target;

    while (cap_left) {
        Square to = pop_lsb(cap_left);
        Square from = to - UpLeft;
        list.add(Move::make(from, to));
    }

    while (cap_right) {
        Square to = pop_lsb(cap_right);
        Square from = to - UpRight;
        list.add(Move::make(from, to));
    }

    // Promotions
    if (promoting) {
        Bitboard promo_push = shift<Up>(promoting) & empty & target;
        Bitboard promo_cap_left = shift<UpLeft>(promoting) & enemies & target;
        Bitboard promo_cap_right = shift<UpRight>(promoting) & enemies & target;

        while (promo_push) {
            Square to = pop_lsb(promo_push);
            Square from = to - Up;
            list.add(Move::make(from, to, MOVE_PROMOTION, QUEEN));
            list.add(Move::make(from, to, MOVE_PROMOTION, ROOK));
            list.add(Move::make(from, to, MOVE_PROMOTION, BISHOP));
            list.add(Move::make(from, to, MOVE_PROMOTION, KNIGHT));
        }

        while (promo_cap_left) {
            Square to = pop_lsb(promo_cap_left);
            Square from = to - UpLeft;
            list.add(Move::make(from, to, MOVE_PROMOTION, QUEEN));
            list.add(Move::make(from, to, MOVE_PROMOTION, ROOK));
            list.add(Move::make(from, to, MOVE_PROMOTION, BISHOP));
            list.add(Move::make(from, to, MOVE_PROMOTION, KNIGHT));
        }

        while (promo_cap_right) {
            Square to = pop_lsb(promo_cap_right);
            Square from = to - UpRight;
            list.add(Move::make(from, to, MOVE_PROMOTION, QUEEN));
            list.add(Move::make(from, to, MOVE_PROMOTION, ROOK));
            list.add(Move::make(from, to, MOVE_PROMOTION, BISHOP));
            list.add(Move::make(from, to, MOVE_PROMOTION, KNIGHT));
        }
    }

    // En passant
    if (pos.ep_square() != SQ_NONE) {
        Bitboard ep_candidates = PawnAttacks[Them][pos.ep_square()] & not_promoting;
        // Note: also check promoting pawns (edge case: promoting pawns can't EP since EP is on rank 6/3)
        while (ep_candidates) {
            Square from = pop_lsb(ep_candidates);
            list.add(Move::make(from, pos.ep_square(), MOVE_EN_PASSANT));
        }
    }
}

void generate_legal(const Position& pos, MoveList& list) {
    list.count = 0;

    Color us = pos.side_to_move();
    Color them = ~us;
    Square ksq = pos.king_sq(us);
    Bitboard checkers = pos.checkers();
    Bitboard occ = pos.pieces();

    // --- King moves (always generated) ---
    Bitboard king_moves = KingAttacks[ksq] & ~pos.pieces(us);
    while (king_moves) {
        Square to = pop_lsb(king_moves);
        // King can't move to attacked square
        Bitboard new_occ = occ ^ square_bb(ksq);
        if (!(pos.attackers_to(to, new_occ) & pos.pieces(them)))
            list.add(Move::make(ksq, to));
    }

    // In double check, only king moves are legal
    if (more_than_one(checkers))
        return;

    // Target squares: if in single check, must block or capture checker
    Bitboard target;
    if (checkers) {
        Square checker_sq = lsb(checkers);
        target = BetweenBB[ksq][checker_sq] | checkers; // block or capture
    } else {
        target = ~pos.pieces(us); // any non-own square
    }

    // Generate piece moves (filtering by target and pin)
    // We generate pseudo-legal then filter via is_legal for pinned pieces

    // --- Pawn moves ---
    if (us == WHITE)
        generate_pawn_moves<WHITE>(pos, list, target);
    else
        generate_pawn_moves<BLACK>(pos, list, target);

    // --- Knight moves ---
    Bitboard knights = pos.pieces(us, KNIGHT) & ~pos.blockers_for_king(us); // pinned knights can never move
    while (knights) {
        Square from = pop_lsb(knights);
        Bitboard moves = KnightAttacks[from] & target;
        while (moves) {
            Square to = pop_lsb(moves);
            list.add(Move::make(from, to));
        }
    }

    // --- Bishop/Queen diagonal moves ---
    Bitboard bishops = pos.pieces(us, BISHOP, QUEEN);
    while (bishops) {
        Square from = pop_lsb(bishops);
        Bitboard moves = bishop_attacks(from, occ) & target;
        while (moves) {
            Square to = pop_lsb(moves);
            list.add(Move::make(from, to));
        }
    }

    // --- Rook/Queen straight moves ---
    Bitboard rooks = pos.pieces(us, ROOK, QUEEN);
    while (rooks) {
        Square from = pop_lsb(rooks);
        Bitboard moves = rook_attacks(from, occ) & target;
        while (moves) {
            Square to = pop_lsb(moves);
            list.add(Move::make(from, to));
        }
    }

    // --- Castling (only when not in check) ---
    if (!checkers) {
        CastlingRight our_oo = king_side(us);
        CastlingRight our_ooo = queen_side(us);

        if (pos.castling() & our_oo) {
            Square rook_sq = (us == WHITE) ? SQ_H1 : SQ_H8;
            Bitboard path = (us == WHITE) ? ((1ULL << SQ_F1) | (1ULL << SQ_G1))
                                         : ((1ULL << SQ_F8) | (1ULL << SQ_G8));
            if (!(occ & path)) {
                // to square for castling move is g1/g8 (but we use the convention: to = towards rook)
                Square to_sq = (us == WHITE) ? SQ_G1 : SQ_G8;
                list.add(Move::make(ksq, to_sq, MOVE_CASTLING));
            }
        }

        if (pos.castling() & our_ooo) {
            Square rook_sq = (us == WHITE) ? SQ_A1 : SQ_A8;
            Bitboard path = (us == WHITE) ? ((1ULL << SQ_D1) | (1ULL << SQ_C1) | (1ULL << SQ_B1))
                                         : ((1ULL << SQ_D8) | (1ULL << SQ_C8) | (1ULL << SQ_B8));
            if (!(occ & path)) {
                Square to_sq = (us == WHITE) ? SQ_C1 : SQ_C8;
                list.add(Move::make(ksq, to_sq, MOVE_CASTLING));
            }
        }
    }

    // --- Filter out illegal moves (pinned pieces + EP legality) ---
    int write = 0;
    for (int i = 0; i < list.count; i++) {
        if (pos.is_legal(list.moves[i])) {
            list.moves[write++] = list.moves[i];
        }
    }
    list.count = write;
}

bool is_checkmate(const Position& pos) {
    if (!pos.checkers()) return false;
    MoveList moves;
    generate_legal(pos, moves);
    return moves.size() == 0;
}

bool is_stalemate(const Position& pos) {
    if (pos.checkers()) return false;
    MoveList moves;
    generate_legal(pos, moves);
    return moves.size() == 0;
}

} // namespace alphaclaude
