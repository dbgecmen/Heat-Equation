# compiler:
CC = g++

# compiler flags:
#  -std=c++14  set C++14 standards
#  -Wall       turns on most, but not all, compiler warnings
#  -pg         write profiling information
CFLAGS  = -std=c++14 -Wall -pg

# the build target executable:
TARGET = heat

all: $(TARGET)

$(TARGET): $(TARGET).cxx
	$(CC) $(CFLAGS) -o $(TARGET) $(TARGET).cxx

clean:
	$(RM) $(TARGET) *~
