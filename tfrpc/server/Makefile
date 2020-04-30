CC = g++
INC = 
LFLAGS = -lssh -ltensorflow
CFLAGS = -g $(INC) -W -Wall

OBJS = yolo_server.o
SRCS = yolo_server.cpp

TARGET = yolo_server

all : $(TARGET)

$(TARGET) : $(OBJS)
	$(CC) $^ -o $@ $(OBJS) $(LFLAGS)

dep :
	gccmakedep $(INC) $(SRCS)

clean :
	rm -rf $(OBJS) $(TARGET) core

# https://www.tensorflow.org/install/lang_c