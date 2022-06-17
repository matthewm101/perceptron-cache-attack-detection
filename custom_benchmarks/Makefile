all: cacheattack idletimer randomupdates
cacheattack: cacheattack.c cacheutils.h
	gcc -std=gnu11 -O3 -o $@ $@.c
idletimer: idletimer.c
	gcc -std=gnu11 -O3 -o $@ $@.c
randomupdates: randomupdates.c cacheutils.h
	gcc -std=gnu11 -O3 -o $@ $@.c