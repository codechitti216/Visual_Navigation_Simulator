.PHONY: clean

LIBOPTS = -I/usr/include/lua5.1 -L/usr/lib/x86_64-linux-gnu -llua5.1
CFLAGS = -fPIC -shared -g
CC = gcc

TORCSctrl.so : TORCSctrl.cpp
	$(CC) $< $(LIBOPTS) $(CFLAGS) -o $@

clean :
	@rm *.so