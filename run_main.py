import datetime
import os
import subprocess

lr = 1e-05
drop = 0.3
max_token_len = 256
hidden_dim = '768'
batch_size = 64
epochs = 4
devices = 8
num_workers = 8
train_size = 0.5
synthetic = True
single_gpu = True


def run_main():
    python_path = '/root/workspace/DlHwVenv/bin/python3.8'
    cmd = [python_path, 'main.py',
           '--lr', str(lr),
           '--drop', str(drop),
           '--max_token_len', str(max_token_len),
           '--hidden_dim', str(hidden_dim),
           '--batch_size', str(batch_size),
           '--epochs', str(epochs),
           '--devices', str(devices),
           '--num_workers', str(num_workers),
           '--single_gpu', str(single_gpu),
           '--synthetic', str(synthetic),
           '--train_size', str(train_size),
           ]

    print(f"\nCurrently Running:\n{' '.join(cmd)}")
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)

    # Generate file name
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H-%M')
    logs_dir = 'logs'
    if not os.path.exists(logs_dir):
        os.makedirs(logs_dir)
    filename = f"{logs_dir}/{timestamp} lr={lr} bs={batch_size} epochs={epochs} drop={drop} num_workers={num_workers} train_s={train_size} synthetic={synthetic}.txt"
    # print("Finished and written to:", filename)
    # print("Output will be written to:", filename)

    # Write output to file
    with open(filename, 'w') as f:
        f.write("=== STDOUT ===\n")
        # Read the output line by line as it becomes available
        while True:
            output = process.stdout.readline()
            if output == '' and process.poll() is not None:
                break
            if output:
                print(output.strip(), flush=True)
                f.write(output)
        # f.write("\n=== STDERR ===\n")
        # err = process.stderr.read()
        # if err:
        #     print(err.strip())
        #     f.write(err)

    print("\nFinished and written to:", filename)

    if process.returncode != 0:
        print("An exception occurred")


if __name__ == "__main__":
    run_main()
