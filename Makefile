OBJS=mnist.jl
CC=julia

run: $(OBJS)
	$(CC) $(OBJS)

clean:
	rm -f *.gz

.PHONY: default all clean
