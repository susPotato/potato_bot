import socket
import time

class SotaSocket:
    def __init__(self, host='localhost', port=11000, bot_name="MK1_BOT"):
        self.host = socket.gethostbyname(socket.gethostname()) if host == 'localhost' else host
        self.port = port
        self.bot_name = bot_name
        self.socket = None

    def connect(self):
        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.connect((self.host, self.port))
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            
            # Register the bot with the server
            init_message = f"name;{self.bot_name}\n"
            self.socket.sendall(init_message.encode('utf-8'))
            print(f"Successfully connected to Sota server at {self.host}:{self.port} as '{self.bot_name}'")
            return True
        except socket.error as e:
            print(f"Failed to connect to Sota server: {e}")
            self.socket = None
            return False

    def send_command(self, command):
        if self.socket:
            try:
                # Make sure the command ends with a newline
                if not command.endswith('\n'):
                    command += '\n'
                self.socket.sendall(command.encode('utf-8'))
            except socket.error as e:
                print(f"Failed to send command: {e}")
        else:
            print("Not connected to Sota server.")

    def receive_message_buffer(self, buffer_size=1024, delimiter="GPTCmd"):
        """
        Receives and buffers data from the socket until a delimiter is found.
        This is a more robust way to handle messages that might be split into multiple packets.
        """
        if not self.socket:
            return None

        all_data = ""
        while True:
            try:
                data = self.socket.recv(buffer_size).decode("utf-8")
                if not data:
                    # Connection closed by server
                    print("Connection closed by the server.")
                    self.socket = None
                    return None
                
                all_data += data
                if delimiter in all_data:
                    # Return the message part before the delimiter
                    message = all_data.split(delimiter)[0]
                    # You might want to handle the part after the delimiter if multiple commands can arrive in one batch
                    return message

            except socket.error as e:
                print(f"Failed to receive message: {e}")
                self.socket = None
                return None

    def close(self):
        if self.socket:
            self.socket.close()
            self.socket = None
            print("Connection to Sota server closed.")
