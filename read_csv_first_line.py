import csv

def read_and_print_line(filename, line_number):
    with open(filename, 'r', newline='') as csvfile:
        reader = csv.reader(csvfile)

        # Iterate until the desired line
        for i, row in enumerate(reader, start=1):
            if i == line_number:
                print(row)
                return

        print(f"File has fewer than {line_number} lines.")

read_and_print_line("data/testing_data.csv", 1000)
