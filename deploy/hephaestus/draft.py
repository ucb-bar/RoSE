x = (7000).to_bytes(4, 'little', signed=False)
print(x)

integer_value = int.from_bytes(x, byteorder='little')
print(integer_value)  # Output: 7000