import os
import zipfile
import random
import string


def generate_new_id(length=16):
    """Generate a random alphanumeric string of a given length."""
    return ''.join(random.choices(string.ascii_letters + string.digits, k=length))


def process_directory(input_dir):
    processed_zip = os.path.join(input_dir, 'processed.zip')

    with zipfile.ZipFile(processed_zip, 'w') as zipf:
        for subdir, dirs, files in os.walk(input_dir):
            if os.path.basename(subdir).startswith('processed'):
                crdfiles = [f for f in files if f.endswith('.crd_body.bin')]
                for crdfile in crdfiles:
                    prefix = crdfile[:-len('.crd_body.bin')]
                    vrt_file = f"{prefix}.vrt_body.bin"

                    if vrt_file in files:
                        new_id = generate_new_id()
                        new_crdfile = f"{new_id}.crd_body.bin"
                        new_vrt_file = f"{new_id}.vrt_body.bin"

                        crd_path = os.path.join(subdir, crdfile)
                        vrt_path = os.path.join(subdir, vrt_file)

                        # Directly write to the zip without any subdirectory structure
                        zipf.write(crd_path, new_crdfile)
                        zipf.write(vrt_path, new_vrt_file)

    print(f"Processed files have been added to {processed_zip}")


# Example usage
input_directory = "D:\\HEAT Dataset\\PRIMER"
process_directory(input_directory)
