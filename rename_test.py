
import os
import re

from params import ParamsInstance as params


history_dir = params.dir_History
dirs_in_history = [x for x in os.listdir(history_dir) if os.path.isdir(os.path.join(history_dir, x))]

model_dir = params.dir_Model
dirs_in_model = [x for x in os.listdir(model_dir) if os.path.isdir(os.path.join(model_dir, x))]

corrects_dirs = [x for x in dirs_in_history if x in dirs_in_model]


for id, elem in enumerate(corrects_dirs):
    print(f"\t{id}: {elem}")
print("\t-1: Exit")

select = int(input("Select directory: "))

if select <= -1 or select >= len(corrects_dirs):
    print("Exit!")

else:

    new_name = input("Insert new name: ")
    confirmation = input(f"Confirm name \"{new_name}\" [y/n]: ")

    if confirmation == "y":
        print(f"Changing name from {corrects_dirs[select]} to {new_name}")

        # print(os.listdir(os.path.join(history_dir, corrects_dirs[select])))
        for name_file in os.listdir(os.path.join(history_dir, corrects_dirs[select])):
            name_splitted = name_file.split("[")
            os.rename(os.path.join(history_dir, corrects_dirs[select], name_file),
                      os.path.join(history_dir, corrects_dirs[select], f"loss_{new_name}_[{name_splitted[1]}") )
            print(f"from {name_file} to loss_{new_name}_[{name_splitted[1]}")
        os.rename(os.path.join(history_dir, corrects_dirs[select]), os.path.join(history_dir, new_name))
        print(f"renamed directory from {corrects_dirs[select]} to new_name")

        # print(os.listdir(os.path.join(model_dir, corrects_dirs[select])))
        for name_file in os.listdir(os.path.join(model_dir, corrects_dirs[select])):
            name_splitted = name_file.split("[")
            os.rename(os.path.join(model_dir, corrects_dirs[select], name_file),
                      os.path.join(model_dir, corrects_dirs[select], f"DeepVO_epoch_{new_name}_[{name_splitted[1]}") )
            print(f"from {name_file} to DeepVO_epoch_{new_name}_[{name_splitted[1]}")
        os.rename(os.path.join(model_dir, corrects_dirs[select]), os.path.join(model_dir, new_name))
        print(f"renamed directory from {corrects_dirs[select]} to new_name")

