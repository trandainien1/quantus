def generate_numbers():
    return list(range(50000))

# Write numbers to a file
with open("numbers.txt", "w") as file:
    for number in generate_numbers():
        file.write(f"{number}\n")