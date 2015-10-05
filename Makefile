CPP_FILES := $(wildcard src/*.cpp)
OBJ_FILES := $(addprefix obj/,$(notdir $(CPP_FILES:.cpp=.o)))
LD_FLAGS := -L./lib -lvl -lmysqlcppconn-static -lboost_regex -lboost_iostreams -lstemmer -lpthread -ldl -ggdb `pkg-config --cflags opencv` `pkg-config --libs opencv`
CC_FLAGS := -I./include -w
OUT := bin/gatorsearch
$(OUT): $(OBJ_FILES)
	mpic++ -o $@ $^ $(LD_FLAGS)
obj/%.o: src/%.cpp include/%.h
	mpic++ $(CC_FLAGS) -c -o $@ $<
clean:
	rm obj/* bin/* 2> /dev/null || true
run:
	mpirun --mca btl_tcp_if_include eth0 -np 4 --hostfile ./mpi_hostfile bin/gatorsearch
