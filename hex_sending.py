from dataclasses import dataclass, field
from typing import Optional
import socket
import asyncio
import pandas as pd
import BAC0
import socket
import time

""" V2 Thermostat Class """

@dataclass
class StatusState:
    switch:        Optional[bool] = None
    mode:          Optional[str]  = None
    temp_set:      Optional[int]  = None
    temp_current:  Optional[int]  = None
    level:         Optional[str]  = None
    mode_lock:     Optional[bool] = None
    temp_lock:     Optional[bool] = None
    keyboard_lock: Optional[bool] = None

    def __repr__(self):
        fields = [f"{name}={getattr(self, name)!r}" for name in self.__dataclass_fields__]
        return f"StatusState({', '.join(fields)})"

    def print_state(self):
        print("StatusState:")
        for name in self.__dataclass_fields__:
            print(f"  {name}: {getattr(self, name)}")

    def update(self, decoded: dict, class_id: int):
        # top-level fields
        self.switch       = decoded["Mode"] != "Off"
        self.mode         = decoded["Mode"].lower()
        self.temp_set     = decoded["Set Temperature"]
        self.temp_current = decoded["Current Temperature"]
        self.level        = decoded["Fan Speed"].lower()

        # nested locks
        locks = decoded["Lock Status"]
        self.keyboard_lock = locks["Keyboard Lock"]
        self.temp_lock     = locks["Temperature Lock"]
        self.mode_lock     = locks["Mode Lock"]
        
        mode_mapping = {"cool": 0, "heat": 1}
        level_mapping = {"low": 0, "middle": 1, "high": 2, "auto": 3}
        
        # Convert mode and level to numeric values
        current_mode = mode_mapping.get(self.mode, -1)  # Default to -1 if value not found
        current_level = level_mapping.get(self.level, -1)  # Default to -1 if value not found

        
        attrs = [
                    (f"Switch_{class_id}",       self.switch ),
                    (f"Mode_{class_id}",         current_mode),
                    (f"Temp_set_{class_id}",     self.temp_set),
                    (f"Level_{class_id}",        current_level),
                    (f"Mode_lock_{class_id}",    self.mode_lock),
                    (f"Temp_lock_{class_id}",    self.temp_lock),
                    (f"Keyboard_lock_{class_id}", self.keyboard_lock)
                ]
        return attrs

@dataclass
class CmdState:
    switch:        Optional[bool] = None
    mode:          Optional[str]  = None
    temp_set:      Optional[int]  = None
    level:         Optional[str]  = None
    mode_lock:     Optional[bool] = None
    temp_lock:     Optional[bool] = None
    keyboard_lock: Optional[bool] = None

    def __repr__(self):
        fields = [f"{name}={getattr(self, name)!r}" for name in self.__dataclass_fields__]
        return f"CmdState({', '.join(fields)})"

    def print_state(self):
        print("CmdState:")
        for name in self.__dataclass_fields__:
            print(f"  {name}: {getattr(self, name)}")

    def update(self, **kwargs):
        for key, value in kwargs.items():
            if key in self.__dataclass_fields__:
                setattr(self, key, value)

@dataclass
class Thermostat:
    ip: str
    device_id: int
    class_id: int #from 1 to n
    conn: Optional[socket.socket] = None  # <-- NEW

    temp_current: StatusState = field(default_factory=StatusState)
    temp_cmd: CmdState = field(default_factory=CmdState)

    def __post_init__(self):
        # Initialize all status and command values to None (BACnet untouched)
        self.temp_current = StatusState()  # All fields default to None
        self.temp_cmd = CmdState()         # All fields default to None

    # Method1: Print out status
    def print_status(self):
        print("=== Thermostat Status ===")
        self.temp_current.print_state()

    # Method2: Print out command values
    def print_cmd(self):
        print("=== Thermostat Command ===")
        self.temp_cmd.print_state()

    # Method3: Assign status value
    def set_status(self, status_dict: dict):
        print("Updating status...")
        self.temp_current.update(**status_dict)
        print("Updated status")

    # Method4: Assign command value
    def set_cmd(self, cmd_dict: dict):
        print("Updating command...")
        self.temp_cmd.update(**cmd_dict)
        print("Updated command")
    
    # Method5: Decode Thermostat Respons
    def decode_thermostat_response(self,response_bytes: bytes):
        if len(response_bytes) < 11:
            return "Invalid response format"

        response_hex = response_bytes.hex().upper()
        response_list = [response_hex[i:i+2] for i in range(0, len(response_hex), 2)]

        device_id = response_list[3]
        mode = response_list[4]
        lock_status_raw = int(response_list[5], 16)
        fan_speed = response_list[6]
        set_temp = int(response_list[7], 16)
        current_temp = int(response_list[8], 16)
        valve_status = response_list[9]

        mode_map = {
            "00": "Off", "01": "heat", "02": "cool", "03": "Auto",
            "04": "Floor Heating", "05": "Rapid Heat", "06": "Ventilation"
        }
        fan_map = {"01": "low", "02": "medium", "03": "high", "04": "uto"}
        valve_map = {"00": "Closed", "01": "Open", "10": "Stopped"}

        keyboard_lock = bool(lock_status_raw & 0b00000001)
        temp_lock = bool(lock_status_raw & 0b00000010)
        mode_lock = bool(lock_status_raw & 0b00000100)

        lock_status_decoded = {
            "Keyboard Lock": keyboard_lock,
            "Temperature Lock": temp_lock,
            "Mode Lock": mode_lock
        }

        return {
            "Device ID": device_id,
            "Mode": mode_map.get(mode, "Unknown"),
            "Lock Status": lock_status_decoded,
            "Fan Speed": fan_map.get(fan_speed, "Unknown"),
            "Set Temperature": set_temp,
            "Current Temperature": current_temp,
            "Valve Status": valve_map.get(valve_status, "Unknown")
        }

    # Method6: Encode Command to Thermostat
    def create_thermostat_command(self, device_id=None, switch=None, current_mode1=None, mode=None, keyboard_lock=None, temp_lock=None, mode_lock=None, fan_speed=None, temperature=None):
        device_id_hex = f"{device_id:02X}"

        if not switch:
            # Switch is OFF → return 01 80 00 ID CHECKSUM
            command = ["01", "80", "00", device_id_hex]
            checksum = sum(int(byte, 16) for byte in command) & 0xFF
            command.append(f"{checksum:02X}")
            return " ".join(command)
        
        if current_mode1 == "off":
            # Mode is "off" → return 01 81 00 ID CHECKSUM
            command = ["01", "81", "00", device_id_hex]
            checksum = sum(int(byte, 16) for byte in command) & 0xFF
            command.append(f"{checksum:02X}")
            return " ".join(command)

        
        command = ["01", "85", "00", f"{device_id:02X}"]

        mode_map = {
            "off": "00", "heat": "01", "cool": "02", "auto": "03",
            "floor_heating": "04", "rapid_heat": "05", "ventilation": "06"
        }
        fan_map = {"low": "01", "medium": "02", "high": "03", "auto": "04"}

        command.append(mode_map.get(mode, "FF"))

        lock_byte = 0
        if keyboard_lock:
            lock_byte |= 0b00000001
        if temp_lock:
            lock_byte |= 0b00000010
        if mode_lock:
            lock_byte |= 0b00000100
        command.append(f"{lock_byte:02X}")

        command.append(fan_map.get(fan_speed, "FF"))
        command.append(f"{temperature:02X}" if temperature is not None else "FF")

        checksum = sum(int(byte, 16) for byte in command) & 0xFF
        command.append(f"{checksum:02X}")

        return " ".join(command)

    # Method7: Handling Initial Connection & Initial Thermostat Status
    def initial_status(self):
        print("Handling Connection with thermostat:", self.ip)
        buffer = b''
        while len(buffer) < 20:
            data = self.conn.recv(1024)
            if not data:
                continue
            print("Initial handshake data skipped:", data.hex())
            buffer += data
        print("Initial handshake data skipped:", buffer.hex())

        #init_command = bytes.fromhex("01 45 00 01 47")
        init_command = bytes([
            0x01,
            0x45,
            0x00,
            self.device_id,
            (0x01 + 0x45 + 0x00 + self.device_id) & 0xFF
        ])
        self.conn.sendall(init_command)
        print("Sent initialization command to conn1:", init_command.hex())
        # Step 3: Wait for "01 85 00" response
        matching_message = None
        get_msg = True
        temp_buffer = b''
        while get_msg:
            data = self.conn.recv(1024)
            print("data received:", data.hex(), "length: ", len(data))
            if not data:
                print("Connection closed while waiting for response.")
                continue
            temp_buffer = data

            while len(temp_buffer) >= 11:
                chunk = temp_buffer[:11]
                temp_buffer = temp_buffer[11:]

                if chunk.startswith(b'\x01\xc5\x00'):
                    get_msg = False
                    matching_message = chunk
                    print("Received target message:", matching_message.hex())

                    decoded = self.decode_thermostat_response(matching_message)
                    print("Decoded thermostat data:", decoded)

                    # Update current variables
                    self.temp_current.update(decoded, self.class_id)
                    print(self.temp_current)
                    break
        print("Finished Inialization of Thermostat Status")

    # Method8: Update Thermostat Status
    def update_status(self):
        print("Handling Connection with thermostat:", self.ip)
        
        data = self.conn.recv(1024)
        #init_command = bytes.fromhex("01 45 00 01 47")
        init_command = bytes([
            0x01,
            0x45,
            0x00,
            self.device_id,
            (0x01 + 0x45 + 0x00 + self.device_id) & 0xFF
        ])
        self.conn.sendall(init_command)
        print("Sent initialization command to conn1:", init_command.hex())
        # Step 3: Wait for "01 85 00" response
        matching_message = None
        get_msg = True
        temp_buffer = b''
        while get_msg:
            data = self.conn.recv(1024)
            print("data received:", data.hex(), "length: ", len(data))
            if not data:
                print("Connection closed while waiting for response.")
                continue
            temp_buffer = data

            while len(temp_buffer) >= 11:
                chunk = temp_buffer[:11]
                temp_buffer = temp_buffer[11:]

                if chunk.startswith(b'\x01\xc5\x00'):
                    get_msg = False
                    matching_message = chunk
                    print("Received target message:", matching_message.hex())

                    decoded = self.decode_thermostat_response(matching_message)
                    print("Decoded thermostat data:", decoded)

                    # Update current variables
                    attrs = self.temp_current.update(decoded,self.class_id)
                    print(self.temp_current)
                    
                    return attrs
                    #break
        print("Finished Inialization of Thermostat Status")

    # Method9: Update and Send Command to Thermostat    
    def send_update_command(self, *, switch, mode, temp_set, level, keyboard_lock, temp_lock, mode_lock):
        mode_mapping_back = {0: "cool", 1:"heat"}
        level_mapping_back = {0: "low", 1: "middle", 2: "high", 3: "auto"}
        boolean_mapping_back = {"inactive": False, "active": True}
        cmd_mode = mode_mapping_back.get(int(mode), -1)  # Default to -1 if value not found
        cmd_level = level_mapping_back.get(int(level), -1)  # Default to -1 if value not found
        cmd_switch = boolean_mapping_back.get(str(switch), True)  # Default to -1 if value not found
        cmd_mode_lock = boolean_mapping_back.get(str(mode_lock), True)  # Default to -1 if value not found
        cmd_temp_lock = boolean_mapping_back.get(str(temp_lock), True)  # Default to -1 if value not found
        cmd_keyboard_lock = boolean_mapping_back.get(str(keyboard_lock), True)  # Default to -1 if value not found
        cmd_temp_set = int(temp_set)
        
        print("thermostat command value after value mapping")
        print(f"get switch: '{cmd_switch}'", type(cmd_switch))
        print(f"get mode: '{cmd_mode}'", type(cmd_mode))
        print(f"get temp_set: '{cmd_temp_set}'", type(cmd_temp_set))
        print(f"get level: '{cmd_level}'",  type(cmd_level))
        print(f"get cmd_mode_lock: '{cmd_mode_lock}'",  type(cmd_mode_lock))
        print(f"get cmd_temp_lock: '{cmd_temp_lock}'",  type(cmd_temp_lock))
        print(f"get cmd_keyboard_lock: '{cmd_keyboard_lock}'",  type(cmd_keyboard_lock))
        
        print("Generate Command for Thermostat")
        send_mode = "off" if not cmd_switch else cmd_mode
        thermo_cmd_str = self.create_thermostat_command(
            device_id=int("01", 16),
            switch = cmd_switch,
            current_mode1=self.temp_current.mode,
            mode=send_mode,
            keyboard_lock=cmd_keyboard_lock,
            temp_lock=cmd_temp_lock,
            mode_lock=cmd_mode_lock,
            fan_speed=cmd_level,
            temperature=cmd_temp_set
        )
        self.conn.sendall(bytes.fromhex(thermo_cmd_str))
        print("Sent command to 1st thermostat:", thermo_cmd_str)
        print("\nCommand send to Thermostat:")
        print(f"  cmd_switch = {cmd_switch}")
        print(f"  cmd_mode = {cmd_mode}")
        print(f"  cmd_temp_set = {cmd_temp_set}")
        print(f"  cmd_level = {cmd_level}")
        print(f"  cmd_mode_lock = {cmd_mode_lock}")
        print(f"  cmd_temp_lock = {cmd_temp_lock}")
        print(f"  cmd_keyboard_lock = {cmd_keyboard_lock}")
        
        self.temp_cmd.update(
            switch=cmd_switch,
            mode=cmd_mode,
            temp_set=cmd_temp_set,
            level=cmd_level,
            keyboard_lock=cmd_keyboard_lock,
            temp_lock=cmd_temp_lock,
            mode_lock=cmd_mode_lock
        )

def main_program():
    #Target Thermostat
    thermostat_list = [
        Thermostat(ip="192.168.12.206", device_id=1, class_id=1)
    ]
    number_ther = 1
    therm = thermostat_list[0]

    ### Please Write Program for Easy Scalling of Thermostat List ###

    """ V3 TCP Connection """

    # 1. Host PC information
    HOST = '192.168.12.195'
    PORT = 8082

    # 2. TCP Socket
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind((HOST, PORT))
    server_socket.listen(100)  # Adjust max queue size for more devices
    print(f"Listening for connections on {HOST}:{PORT}...")

    # 3. Thermostat dictionary for fast IP-to-thermostat lookup & Connection
    thermostat_dict = {t.ip: t for t in thermostat_list}
    ALLOWED_IPS = set(thermostat_dict.keys())  # Faster lookup than list

    # 4. TCP Connection with All Target Thermostat 
    cnt = 0
    while cnt < number_ther: #continue to finding new connection untill all target thermostat are connected
        conn, addr = server_socket.accept()
        client_ip = addr[0]
        if client_ip in ALLOWED_IPS:
            print(f"Accepted connection from allowed device: {client_ip}")
            cnt+=1
            thermostat = thermostat_dict[client_ip]
            thermostat.conn = conn
            print(f"Stored connection in Thermostat {thermostat.device_id} ({client_ip})")
        else:
            print(f"Ignored unauthorized connection from: {client_ip}")
            conn.close()

    """ V4 Thermostat Initialization """

    # 5. Get the thermostat status
    for therm in thermostat_list:
        therm.initial_status()

    ####################turn off thermostat######################
    # 设置一个变量
    therm1 = thermostat_list[0]
    print(therm1.conn)
    init_command = bytes([
            0x01,
            0x80,
            0x00,
            0x01,
            0x82
        ])# turn off
    if therm1.conn is not None:
        therm1.conn.sendall(init_command)
    else:
        print("Warning: therm1.conn is None, cannot send init_command.")
    time.sleep(10)
    ####################turn on thermostat######################
    therm1 = thermostat_list[0]
    init_command = bytes([
            0x01,
            0x81,
            0x00,
            0x01,
            0x83
        ])# turn on
    if therm1.conn is not None:
        therm1.conn.sendall(init_command)
    else:
        print("Warning: therm.conn is None, cannot send init_command.")
    
    """# 7. Get Command From DDC
    print(f"for Commnad of thermostat Class ID: {class_id}")
    #1) Read all cmd values into a dict
    attrs = [
        ("switch",    f"Switch_{class_id}"),
        ("mode",      f"Mode_{class_id}"),
        ("temp_set",  f"Temp_set_{class_id}"),
        ("level",     f"Level_{class_id}"),
        ("mode_lock", f"Mode_lock_{class_id}"),
        ("temp_lock", f"Temp_lock_{class_id}"),
        ("keyboard_lock", f"Keyboard_lock_{class_id}")
    ]

    cmds = {}
    for key, prop in attrs:
        val = await test_device[prop].read_property("presentValue")
        print(f"get {key}: ", val)
        await asyncio.sleep(1)
        cmds[key] = val

    # 8. Send Command to Thermostat
    therm = thermostat_class_dict[class_id]
    therm.send_update_command(**cmds)
    attrs2 = therm.update_status()"""
    
                
if __name__ == "__main__":
    main_program()