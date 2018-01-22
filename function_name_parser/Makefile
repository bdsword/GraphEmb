all: parser.o lex.o main.o
	gcc -o extract_func_name parser.o lex.o main.o -lfl
main.o: main.c
	gcc -o main.o -c main.c
parser.o: parser.tab.c
	gcc -o parser.o -c parser.tab.c
parser.tab.c: parser.y
	bison -d parser.y
lex.o: lex.yy.c
	gcc -o lex.o -c lex.yy.c
lex.yy.c: parser.l
	flex parser.l
clean:
	rm *.o extract_func_name parser.tab.c lex.yy.c

