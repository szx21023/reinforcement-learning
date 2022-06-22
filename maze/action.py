class Action:
    def __init__(self, name: str, code: int, direction: list):
        self.name = name
        self.code = code
        self.direction = direction

ACTIONS = [
    Action('up', 0, [-1, 0]),
    Action('down', 1, [1, 0]),
    Action('left', 2, [0, -1]),
    Action('right', 3, [0, 1])
]