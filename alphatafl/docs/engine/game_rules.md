---
title: Hnefatafl Game Rules
---

# Hnefatafl Game Rules

This document describes the rules implemented in the AlphaTafl engine, based on the standard 11x11 board version.

## 1. The Objective

Hnefatafl is an asymmetrical game with two sides:

*   **The Defenders (White)**: Goal is to help the King reach one of the four corner squares.
*   **The Attackers (Black)**: Goal is to surround and capture the King.

## 2. Board and Setup

*   **Board**: 11x11 grid.
*   **Pieces**: 1 King, 12 Defenders, 24 Attackers.
*   **Initial Layout**:
    *   King on the central **Throne**.
    *   12 Defenders surrounding the King.
    *   24 Attackers in four groups of six on the edges.
*   **Restricted Squares**:
    *   The **Throne** (center) and the four **Corners**.
    *   Only the King can land on these squares.
    *   Other pieces can pass *over* the Throne if empty.

## 3. Movement

*   All pieces move orthogonally (up, down, left, right).
*   Pieces can move any number of empty spaces in a straight line (like a Rook in Chess).
*   No jumping over pieces.
*   **Attackers move first.**

## 4. Capturing Regular Pieces

*   **Custodian Capture**: A piece is captured when it is "sandwiched" between two opposing pieces on opposite sides (horizontally or vertically).
*   **Active Capture**: Captures only occur on an active move. Moving between two enemies does *not* result in capture.
*   **Hostile Squares**: Corner squares and the empty Throne act as hostile squares. A piece can be captured by being sandwiched between an enemy and a hostile square.
*   **Multiple Captures**: Possible if a single move creates multiple sandwiches.

## 5. Capturing the King

*   **In the Open**: Must be surrounded on all **four** orthogonal sides.
*   **Against an Edge or Throne**: Must be surrounded on the remaining **three** sides.
*   **Win Condition**: Captured King = Attacker victory.

## 6. Winning the Game

*   **Defenders Win**: King reaches a corner square.
*   **Attackers Win**: King is captured.

---
Source: [game_rules.md](../../game_rules.md)
