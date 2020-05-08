cflags=-lrealm -lpthread -ldl -lrt -g

test: test.cc realm_wrapper.o
	g++ test.cc realm_wrapper.o ${cflags}  -o test

realm_wrapper.o: realm_wrapper.cc
	g++ -g -c realm_wrapper.cc 

