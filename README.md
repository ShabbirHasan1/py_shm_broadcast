# py-shm-broadcast

A message queue that uses shared memory to communicate between processes.

This class provides a message queue that uses shared memory to communicate
between processes. It is designed to be used in a distributed computing
environment where multiple processes need to communicate with each other.
The shared memory ring buffer is used to store the data that is being
communicated, and the publish-subscribe pattern is used to send messages
between the processes.

The shared memory ring buffer is implemented using the `multiprocessing`
module, which allows multiple processes to access the same shared memory
buffer. The shared memory ring buffer is created using the `multiprocessing`
module, and the publish-subscribe pattern is implemented using the `zmq`
library.

The shared memory ring buffer is used to store the data that is being
communicated. The data is stored in chunks, and each chunk is associated
with a flag that indicates whether the chunk has been read or not. The
shared memory ring buffer is divided into two parts: the metadata part and
the data part. The metadata part contains the flags that indicate whether
the chunk has been read or not. The data part contains the actual data.

The publish-subscribe pattern is used to send messages between the processes.
Each process has a publish-subscribe socket that is used to send messages to
all other processes. The publish-subscribe socket is used to send messages
to all other processes, and the subscribe-publish socket is used to receive
messages from all other processes.

The shared memory ring buffer is divided into two parts: the metadata part
and the data part. The metadata part contains the flags that indicate whether
the chunk has been read or not. The data part contains the actual data.

The shared memory ring buffer is divided into two parts: the metadata part
and the data part. The metadata part contains the flags that indicate whether
the chunk has been read or not. The data part contains the actual data.
