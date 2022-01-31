
SRC=main.cc
OUT=out

all: compile run

compile:
	g++ $(SRC) -o $(OUT)

run:
	./$(OUT)
