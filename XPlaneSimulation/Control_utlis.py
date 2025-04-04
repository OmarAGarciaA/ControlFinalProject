# -*- coding: utf-8 -*-
#!/usr/bin/env python3
#################################################################################
#                               30/01/2023
#author: Omar García
#github: https://github.com/OmarGAlcantara/MIL-Nengo-XPlane

# This is a modified version of the XPC file from XPlaneConnect necessary for handling the UDP communication
# with XPlane 11 via XPC
#################################################################################

import sys
import time
import datetime
import rospy
from std_msgs.msg import String, Bool, UInt16, Float64
from geometry_msgs.msg import TwistStamped, Twist, PoseStamped, Pose
from sensor_msgs.msg import Imu
from nav_msgs.msg import Odometry

import struct 
import socket

import xplane_ros.msg as xplane_msgs
import rosplane_msgs.msg as rosplane_msgs

class XPlaneConnect(object):
    """XPlaneConnect (XPC) facilitates communication to and from the XPCPlugin."""
    socket = None

    # Basic Functions
    def __init__(self, xpHost='localhost', xpPort=49000, port=0, timeout=100):
        """Sets up a new connection to an X-Plane Connect plugin running in X-Plane.

            Args:
              xpHost: The hostname of the machine running X-Plane.
              xpPort: The port on which the XPC plugin is listening. Usually 49007.
              port: The port which will be used to send and receive data.
              timeout: The period (in milliseconds) after which read attempts will fail.
        """

        # Validate parameters
        xpIP = None
        try:
            xpIP = socket.gethostbyname(xpHost)
        except:
            raise ValueError("Unable to resolve xpHost.")

        if xpPort < 0 or xpPort > 65535:
            raise ValueError("The specified X-Plane port is not a valid port number.")
        if port < 0 or port > 65535:
            raise ValueError("The specified port is not a valid port number.")
        if timeout < 0:
            raise ValueError("timeout must be non-negative.")

        # Setup XPlane IP and port
        self.xpDst = (xpIP, xpPort)

        # Create and bind socket
        clientAddr = ("0.0.0.0", port)
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
        self.socket.bind(clientAddr)
        timeout /= 1000.0
        self.socket.settimeout(timeout)

    def __del__(self):
        self.close()

    # Define __enter__ and __exit__ to support the `with` construct.
    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self.close()

    def close(self):
        """Closes the specified connection and releases resources associated with it."""
        if self.socket is not None:
            self.socket.close()
            self.socket = None

    def sendUDP(self, buffer):
        """Sends a message over the underlying UDP socket."""
        # Preconditions
        if(len(buffer) == 0):
            raise ValueError("sendUDP: buffer is empty.")

        self.socket.sendto(buffer, 0, self.xpDst)
	#socket.sendto(bytes, flags, address)

    def readUDP(self):
        """Reads a message from the underlying UDP socket."""
        return self.socket.recv(16384)

    # Configuration
    def setCONN(self, port):
        """Sets the port on which the client sends and receives data.

            Args:
              port: The new port to use.
        """

        #Validate parameters
        if port < 0 or port > 65535:
            raise ValueError("The specified port is not a valid port number.")

        #Send command
        buffer = struct.pack(b"<4sxH", b"CONN", port)
        self.sendUDP(buffer)

        #Rebind socket
        clientAddr = ("0.0.0.0", port)
        timeout = self.socket.gettimeout()
        self.socket.close()
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
        self.socket.bind(clientAddr)
        self.socket.settimeout(timeout)

        #Read response
        buffer = self.socket.recv(1024)

    def pauseSim(self, pause):
        """Pauses or un-pauses the physics simulation engine in X-Plane.

            Args:
              pause: True to pause the simulation; False to resume.
        """
        pause = int(pause)
        if pause < 0 or pause > 2:
            raise ValueError("Invalid argument for pause command.")

        buffer = struct.pack(b"<4sxB", b"SIMU", pause)
        self.sendUDP(buffer)

    # X-Plane UDP Data
    def readDATA(self):
        """Reads X-Plane data.

            Returns: A 2 dimensional array containing 0 or more rows of data. Each array
              in the result will have 9 elements, the first of which is the row number which
              that array represents data for, and the rest of which are the data elements in
              that row.
        """
        buffer = self.readUDP()
        if len(buffer) < 6:
            return None
        rows = (len(buffer) - 5) // 36
        data = []
        for i in range(rows):
            data.append(struct.unpack_from(b"9f", buffer, 5 + 36*i))
        return data

    def sendDATA(self, data):
        """Sends X-Plane data over the underlying UDP socket.

            Args:
              data: An array of values representing data rows to be set. Each array in `data`
                should have 9 elements, the first of which is a row number in the range (0-134),
                and the rest of which are the values to set for that data row.
        """
        if len(data) > 134:
            raise ValueError("Too many rows in data.")

        buffer = struct.pack(b"<4sx", b"DATA") # b is for integer
							
	# So overall, the line of code you provided packs the bytes corresponding to the string "DATA" 	
	#along with a single pad byte, into a 5-byte buffer in little-endian byte order. 4 bytes of 	
	#string +  1 single padding byte

	#struct.pack(format, v1, v2, ...)
	#Return a bytes object containing the values v1, v2, … packed according to the format string format. 		#The arguments must match the values required by the format exactly.	

        for row in data:
            if len(row) != 9:
                raise ValueError("Row does not contain exactly 9 values. <" + str(row) + ">")
            buffer += struct.pack(b"<I8f", *row)
        self.sendUDP(buffer)

		#Therefore, the format string "I8f" describes a data structure that contains a 32-bit unsigned 			integer followed by eight 32-bit floating-point numbers

    # Position
    def getPOSI(self, ac=0):
        """Gets position information for the specified aircraft.

        Args:
          ac: The aircraft to get the position of. 0 is the main/player aircraft.
        """
        # Send request
        buffer = struct.pack(b"<4sxB", b"GETP", ac)
        self.sendUDP(buffer)

        # Read response
        resultBuf = self.readUDP()
        if len(resultBuf) == 34:
            result = struct.unpack(b"<4sxBfffffff", resultBuf)
        elif len(resultBuf) == 46:
            result = struct.unpack(b"<4sxBdddffff", resultBuf)
        else:
            raise ValueError("Unexpected response length.")

        if result[0] != b"POSI":
            raise ValueError("Unexpected header: " + result[0])

        # Drop the header & ac from the return value
        return result[2:]

    def sendPOSI(self, values, ac=0):
        """Sets position information on the specified aircraft.

            Args:
              values: The position values to set. `values` is a array containing up to
                7 elements. If less than 7 elements are specified or any elment is set to `-998`,
                those values will not be changed. The elements in `values` corespond to the
                following:
                  * Latitude (deg)
                  * Longitude (deg)
                  * Altitude (m above MSL)
                  * Pitch (deg)
                  * Roll (deg)
                  * True Heading (deg)
                  * Gear (0=up, 1=down)
              ac: The aircraft to set the position of. 0 is the main/player aircraft.
        """
    
        # Preconditions
        if len(values) < 1 or len(values) > 7:
            raise ValueError("Must have between 0 and 7 items in values.")
        if ac < 0 or ac > 20:
            raise ValueError("Aircraft number must be between 0 and 20.")

        # Pack message
        buffer = struct.pack(b"<4sxB", b"POSI", ac)
        for i in range(7):
            val = -998
            if i < len(values):
                val = values[i]
            if i < 3:
                buffer += struct.pack(b"<d", val)
            else:
                buffer += struct.pack(b"<f", val)

        # Send
        self.sendUDP(buffer)

    # Controls
    def getCTRL(self, ac=0):
        """Gets the control surface information for the specified aircraft.

        Args:
          ac: The aircraft to get the control surfaces of. 0 is the main/player aircraft.
        """
        # Send request
        buffer = struct.pack(b"<4sxB", b"GETC", ac)
        self.sendUDP(buffer)

        # Read response
        resultBuf = self.readUDP()
        if len(resultBuf) != 31:
            raise ValueError("Unexpected response length.")

        result = struct.unpack(b"<4sxffffbfBf", resultBuf)
        if result[0] != b"CTRL":
            raise ValueError("Unexpected header: " + result[0])

        # Drop the header from the return value
        result =result[1:7] + result[8:]
        return result

    def sendCTRL(self, values, ac=0):
        """Sets control surface information on the specified aircraft.

            Args:
              values: The control surface values to set. `values` is a array containing up to
                6 elements. If less than 6 elements are specified or any elment is set to `-998`,
                those values will not be changed. The elements in `values` corespond to the
                following:
                  * Latitudinal Stick [-1,1]
                  * Longitudinal Stick [-1,1]
                  * Rudder Pedals [-1, 1]
                  * Throttle [-1, 1]
                  * Gear (0=up, 1=down)
                  * Flaps [0, 1]
                  * Speedbrakes [-0.5, 1.5]
              ac: The aircraft to set the control surfaces of. 0 is the main/player aircraft.
        """
        # Preconditions
        if len(values) < 1 or len(values) > 7:
            raise ValueError("Must have between 0 and 6 items in values.")
        if ac < 0 or ac > 20:
            raise ValueError("Aircraft number must be between 0 and 20.")

        # Pack message
        buffer = struct.pack(b"<4sx", b"CTRL")
        for i in range(6):
            val = -998
            if i < len(values):
                val = values[i]
            if i == 4:
                val = -1 if (abs(val + 998) < 1e-4) else val
                buffer += struct.pack(b"b", int(val))
            else:
                buffer += struct.pack(b"<f", val)

        buffer += struct.pack(b"B", ac)
        if len(values) == 7:
            buffer += struct.pack(b"<f", values[6])

        # Send
        self.sendUDP(buffer)

    # DREF Manipulation
    def sendDREF(self, dref, values):
        """Sets the specified dataref to the specified value.

            Args:
              dref: The name of the datarefs to set.
              values: Either a scalar value or a sequence of values.
        """
        
        self.sendDREFs([dref], [values])


    def sendDREFs(self, drefs, values):
        """Sets the specified datarefs to the specified values.

        Args:
        drefs: A list of names of the datarefs to set.
        values: A list of scalar or vector values to set.
        """
        if len(drefs) != len(values):
            raise ValueError("drefs and values must have the same number of elements.")

        buffer = struct.pack("<4sx", "DREF")  # 'DREF' header + NULL terminator

        for i in range(len(drefs)):
            dref = drefs[i]
            value = values[i]

            # Validate dref
            if not isinstance(dref, str) or len(dref) == 0 or len(dref) > 500:
                raise ValueError("dref must be a non-empty string less than 500 characters.")

            # Validate value
            if value is None:
                raise ValueError("value must be a scalar or sequence of floats.")

            # Ensure single-element lists are converted to a float
            if isinstance(value, list) and len(value) == 1:
                value = value[0]

            # Ensure the value is properly formatted as a float
            if not isinstance(value, (int, float)):
                raise TypeError("value must be a float or a list of floats.")

            # Pack message
            fmt = "<f{0:d}s".format(500 - len(dref))  # Pad dataref to 500 bytes
            packed_dref = dref.encode() + b"\x00" * (500 - len(dref))  # Null-pad

            buffer += struct.pack("<f", float(value))  # Send value as a float
            buffer += packed_dref[:500]  # Trim to exactly 500 bytes

        # Debugging output
        # print("Sending DREF packet to X-Plane")
        # print("DREFs:", drefs)
        # print("Values:", values)
        # print("Buffer size:", len(buffer))  # Should be 509 bytes

        # Send packet
        self.sendUDP(buffer)

# --------------------------------------------------------------------------
    def sendDREFstest(self, drefs, values):
        """Sets the specified datarefs to the specified values.

        Args:
            drefs: A list of names of the datarefs to set.
            values: A list of scalar or vector values to set.
        """
        if len(drefs) != len(values):
            raise ValueError("drefs and values must have the same number of elements.")

        buffer = struct.pack("<4sx", b"DREF")  # 'DREF' header + NULL terminator

        for i in range(len(drefs)):
            dref = drefs[i]
            value = values[i]

            # Validate dref
            if not isinstance(dref, str) or len(dref) == 0 or len(dref) > 500:
                raise ValueError("dref must be a non-empty string less than 500 characters.")

            # Validate value
            if value is None:
                raise ValueError("value must be a scalar or sequence of floats.")

            # Ensure value is a list of floats
            if isinstance(value, (int, float)):
                value = [float(value)]  # Convert scalar to list
            elif isinstance(value, list) and all(isinstance(v, (int, float)) for v in value):
                value = [float(v) for v in value]  # Ensure list contains floats
            else:
                raise TypeError("value must be a float or a list of floats.")

            # Determine how many floats are being sent
            num_floats = len(value)
            fmt = "<{}f".format(num_floats)  # Python 2.7-compatible string formatting

            # Adjust DREF padding to fit exactly 509 bytes
            # DREF header = 4 bytes, quaternion = 16 bytes, remaining 489 bytes for DREF string
            dref_padding_size = 500 - (4 * num_floats)  # Correct padding size based on num_floats
            packed_dref = dref.encode('utf-8') + b"\x00" * dref_padding_size  # Null-pad dynamically

            buffer += struct.pack(fmt, *value)  # Pack float values
            buffer += packed_dref[:dref_padding_size]  # Trim to the correct size

        # Debugging output
        print("Sending DREF packet to X-Plane")
        print("DREFs:", drefs)
        print("Values:", values)
        print("Buffer size:", len(buffer))  # Should always be 509 bytes

        # Send packet
        self.sendUDP(buffer)
#----------------------------------------------------------------------------


    def getDREF(self, dref):
        """Gets the value of an X-Plane dataref.

            Args:
              dref: The name of the dataref to get.

            Returns: A sequence of data representing the values of the requested dataref.
        """
        return self.getDREFs([dref])[0]

    def getDREFs(self, drefs):
        """Gets the value of one or more X-Plane datarefs.

            Args:
              drefs: The names of the datarefs to get.

            Returns: A multidimensional sequence of data representing the values of the requested
             datarefs.
        """
        # Send request
        buffer = struct.pack(b"<4sxB", b"GETD", len(drefs))
        for dref in drefs:
            fmt = "<B{0:d}s".format(len(dref))
            buffer += struct.pack(fmt.encode(), len(dref), dref.encode())
        self.sendUDP(buffer)

        # Read and parse response
        buffer = self.readUDP()
        resultCount = struct.unpack_from(b"B", buffer, 5)[0]
        offset = 6
        result = []
        for i in range(resultCount):
            rowLen = struct.unpack_from(b"B", buffer, offset)[0]
            offset += 1
            fmt = "<{0:d}f".format(rowLen)
            row = struct.unpack_from(fmt.encode(), buffer, offset)
            result.append(row)
            offset += rowLen * 4
        return result

    # Drawing
    def sendTEXT(self, msg, x=-1, y=-1):
        """Sets a message that X-Plane will display on the screen.

            Args:
              msg: The string to display on the screen
              x: The distance in pixels from the left edge of the screen to display the
                 message. A value of -1 indicates that the default horizontal position should
                 be used.
              y: The distance in pixels from the bottom edge of the screen to display the
                 message. A value of -1 indicates that the default vertical position should be
                 used.
        """
        if y < -1:
            raise ValueError("y must be greater than or equal to -1.")

        if msg == None:
            msg = ""

        msgLen = len(msg)

        # TODO: Multiple byte conversions
        buffer = struct.pack(b"<4sxiiB" + (str(msgLen) + "s").encode(), b"TEXT", x, y, msgLen, msg.encode())
        self.sendUDP(buffer)

    def sendVIEW(self, view):
        """Sets the camera view in X-Plane

            Args:
              view: The view to use. The ViewType class provides named constants
                    for known views.
        """
        # Preconditions
        if view < ViewType.Forwards or view > ViewType.FullscreenNoHud:
            raise ValueError("Unknown view command.")

        # Pack buffer
        buffer = struct.pack(b"<4sxi", b"VIEW", view)

        # Send message
        self.sendUDP(buffer)

    def sendWYPT(self, op, points):
        """Adds, removes, or clears waypoints. Waypoints are three dimensional points on or
           above the Earth's surface that are represented visually in the simulator. Each
           point consists of a latitude and longitude expressed in fractional degrees and
           an altitude expressed as meters above sea level.

            Args:
              op: The operation to perform. Pass `1` to add waypoints,
                `2` to remove waypoints, and `3` to clear all waypoints.
              points: A sequence of floating point values representing latitude, longitude, and
                altitude triples. The length of this array should always be divisible by 3.
        """
        if op < 1 or op > 3:
            raise ValueError("Invalid operation specified.")
        if len(points) % 3 != 0:
            raise ValueError("Invalid points. Points should be divisible by 3.")
        if len(points) / 3 > 255:
            raise ValueError("Too many points. You can only send 255 points at a time.")

        if op == 3:
            buffer = struct.pack(b"<4sxBB", b"WYPT", 3, 0)
        else:
            buffer = struct.pack(("<4sxBB" + str(len(points)) + "f").encode(), b"WYPT", op, len(points), *points)
        self.sendUDP(buffer)

class ViewType(object):
    Forwards = 73
    Down = 74
    Left = 75
    Right = 76
    Back = 77
    Tower = 78
    Runway = 79
    Chase = 80
    Follow = 81
    FollowWithPanel = 82
    Spot = 83
    FullscreenWithHud = 84
    FullscreenNoHud = 85


def DecodeDataMessage(message):

  # Buscar esta funcion en google para endender como vienen ordenados los datos


  # Message consists of 4 byte type and 8 times a 4byte float value.
  # Write the results in a python dict. 
  values = {}
  #p = {}
  typelen = 4
  type = int.from_bytes(message[0:typelen], byteorder='little')
  data = message[typelen:]
  dataFLOATS = struct.unpack("<ffffffff",data)

  if type == 0:
    values["FPS"] =int(dataFLOATS[0])
 # elif type == 3:
   # values["Speed"]=int(dataFLOATS[0])
 # elif type == 14:
  #  values["Gear"]=int(dataFLOATS[0])
  #  values["Brakes"]=round(dataFLOATS[1],3)
  elif type == 16:
    values["Q"]=dataFLOATS[0]
    values["P"]=dataFLOATS[1]
    values["R"]=dataFLOATS[2]
  elif type == 17:
    values["pitch"]=dataFLOATS[0]
    values["roll"]=dataFLOATS[1]
    values["heading"]=dataFLOATS[2]
    values["heading2"]=dataFLOATS[3]
  elif type == 20:
    values["latitude"]=dataFLOATS[0]
    values["longitude"]=dataFLOATS[1]
    values["Alt"]=int(dataFLOATS[2])
    values["altitude AGL"]=dataFLOATS[3]
    values["altitude 2"]=dataFLOATS[4]
    values["altitude 3"]=dataFLOATS[5]
  elif type == 25:
    values["Throttle Comm"]=int(round(dataFLOATS[0], 3) * 100)
  #elif type == 37:
   # values["RPM"]=int(round(dataFLOATS[0], 1))
  else:
    print("  Type ", type, " other not implemented: ",dataFLOATS)
  return values

def DecodePacket(data):
  # Packet consists of 5 byte header and multiple messages. 
  #debug by uncommenting next line
  #print("raw data-> %s" % data, end="\r", flush=False)

  valuesout = {}
  headerlen = 5
  header = data[0:headerlen]
  messages = data[headerlen:]
  if(header==b'DATA*'):
    # Divide into 36 byte messages
    messagelen = 36
    #print("Processing")
    for i in range(0,int((len(messages))/messagelen)):
      message = messages[(i*messagelen) : ((i+1)*messagelen)]
      values = DecodeDataMessage(message)
      valuesout.update( values )
  else:
    print("THIS packet type not implemented. ")
    #print("  Header: ", header)
    #print("  Data-> ", messages)

  return valuesout


