
SRC=main.cc
OUT=out
STD=-std=c++20
OPT_LEVEL=-O0

all: compile run

compile:
	g++ $(SRC) $(STD) $(OPT_LEVEL) -o $(OUT)

run:
	./$(OUT)
