import os
import shutil

source_mnist = "MNIST results"
source_cifar = "cifar10_results_backup"
dest_folder = "Final_Merged_Results"

def merge_folders():
    print(f"Creating {dest_folder}...")
    os.makedirs(dest_folder, exist_ok=True)
    
    print(f"Copying {source_mnist}...")
    if os.path.exists(source_mnist):
        shutil.copytree(source_mnist, os.path.join(dest_folder, "MNIST"), dirs_exist_ok=True)
    else:
        print(f"Warning: {source_mnist} not found.")

    print(f"Copying {source_cifar}...")
    if os.path.exists(source_cifar):
        shutil.copytree(source_cifar, os.path.join(dest_folder, "CIFAR-10"), dirs_exist_ok=True)
    else:
        print(f"Warning: {source_cifar} not found.")
        
    print(f"\n✅ All results successfully merged into '{dest_folder}' folder!")

if __name__ == "__main__":
    merge_folders()
