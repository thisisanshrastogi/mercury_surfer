import os 

folder = './reviews'

total_lines = 0

for filename in os.listdir(folder):
    if(filename.endswith(".csv")):
        file_path = os.path.join(folder,filename)
        with open(file_path,"r",encoding="utf-8") as file:
            line_count = sum(1 for line in file) -1

        total_lines +=line_count
        print(f"{filename} has {line_count} reviews") 

print(f"Total reviews {total_lines}")