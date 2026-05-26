class MockEncoder:
    def __init__(self):
        self.pos_estimate = 0.0
        self.vel_estimate = 0.0

class MockController:
    def __init__(self):
        self.input_torque = 0.0

class MockAxis:
    def __init__(self, motor_id):
        self.motor_id = motor_id
        self.encoder = MockEncoder()
        self.controller = MockController()
        self.requested_state = None

    def clear_errors(self):
        print(f"[MOCK] Motor {self.motor_id}: errors cleared")
