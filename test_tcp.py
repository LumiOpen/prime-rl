import socket
s = socket.socket()
address = '10.32.16.239'
port = 13346  # port number is a number, not string
try:
    s.connect((address, port)) 
    # originally, it was 
    # except Exception, e: 
    # but this syntax is not supported anymore.
    print("successfully connected to %s:%d" % (address, port))
except Exception as e: 
    print("something's wrong with %s:%d. Exception is %s" % (address, port, e))
finally:
    s.close()