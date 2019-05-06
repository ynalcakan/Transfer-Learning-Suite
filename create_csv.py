import os

with open("data.csv", "w+") as f:
    with os.scandir("/home/yagiz/funghi/funghi/funghi_images/") as entries:
        for entry in entries:
            filename = entry.name
            if "e_" in filename:
                csv_line = "{},0".format(filename)
            elif "i_" in filename:
                csv_line = "{},1".format(filename)
            else:
                csv_line = "{},2".format(filename)

            f.write(csv_line + "\n")
    
