#pragma once

#include "types.h"
#include "position.h"

namespace alphaclaude {

// Generate all legal moves
void generate_legal(const Position& pos, MoveList& list);

// Check if position is checkmate or stalemate
bool is_checkmate(const Position& pos);
bool is_stalemate(const Position& pos);

} // namespace alphaclaude
