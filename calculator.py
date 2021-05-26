class Calculator():
    def __init__(self):
        self.a = None
        self.b = None

    def add_number(self, number):
        if self.a is None:
            self.a = number
        elif self.b is None:
            self.b = number
        else:
            self.a = None
            self.b = None

    @property
    def result(self):
        if self.a is None or self.b is None:
            return None

        return self.a + self.b

    def __str__(self):
        a = "?" if self.a is None else str(self.a)
        b = "?" if self.b is None else str(self.b)
        c = "?" if self.a is None or self.b is None else str(self.result)

        return f"{a} + {b} = {c}"
