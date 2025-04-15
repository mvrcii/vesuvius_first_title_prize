import argparse
import concurrent.futures
import glob
import os
import signal
from functools import partial
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup
from rich.console import Console
from rich.progress import Progress, TextColumn, BarColumn, TaskProgressColumn

console = Console()
interrupted = False


def signal_handler(sig, frame):
    global interrupted
    interrupted = True


signal.signal(signal.SIGINT, signal_handler)


def get_all_links(url, base_url, file_list=None):
    if file_list is None:
        file_list = []
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        for link in soup.find_all('a'):
            if interrupted:
                return file_list
            href = link.get('href')
            if not href or href in ['../', '/'] or '?C=' in href:
                continue
            file_url = urljoin(url, href)
            if href.endswith('/'):
                get_all_links(file_url, base_url, file_list)
            else:
                rel_path = os.path.relpath(urlparse(file_url).path, urlparse(base_url).path)
                file_list.append((file_url, rel_path))
    except Exception as e:
        console.log(f"[red]Error fetching {url}: {e}[/red]")
    return file_list


def download_file(file_info, target_dir, progress):
    global interrupted
    if interrupted:
        return False, "", "interrupted", 0

    file_url, rel_path = file_info
    file_path = os.path.join(target_dir, rel_path)
    partial_path = file_path + ".partial"
    filename = os.path.basename(file_path)

    if os.path.exists(file_path) and os.path.getsize(file_path) > 0:
        return True, filename, "skipped", 0

    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    try:
        response = requests.get(file_url, stream=True, timeout=30)
        response.raise_for_status()
        total_size = int(response.headers.get('content-length', 0))
        block_size = 8192
        downloaded = 0
        task_id = progress.add_task(filename, total=total_size)
        with open(partial_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=block_size):
                if interrupted:
                    progress.update(task_id, visible=False)
                    return False, filename, "interrupted", downloaded
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    progress.update(task_id, completed=downloaded)
        if os.path.exists(partial_path) and os.path.getsize(partial_path) > 0:
            os.replace(partial_path, file_path)
            progress.update(task_id, visible=False)
            return True, filename, "downloaded", total_size
        else:
            progress.update(task_id, visible=False)
            return False, filename, "failed", 0
    except Exception as e:
        progress.update(task_id, visible=False)
        return False, filename, f"error: {str(e)[:40]}", 0


def cleanup_partial_files(target_dir):
    for p_file in glob.glob(os.path.join(target_dir, "**/*.partial"), recursive=True):
        try:
            os.remove(p_file)
        except OSError:
            pass


def download_files_parallel(url, target_dir, max_workers=5):
    global interrupted
    os.makedirs(target_dir, exist_ok=True)
    cleanup_partial_files(target_dir)

    file_list = get_all_links(url, url)
    console.log(f"Found {len(file_list)} files to download. Starting ..")
    if not file_list:
        return

    completed = successful = skipped = failed = bytes_downloaded = 0
    with Progress(
            TextColumn("{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console,
            transient=True
    ) as progress:
        status_task = progress.add_task("Processing files", total=len(file_list))
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            download_func = partial(download_file, target_dir=target_dir, progress=progress)
            futures = {executor.submit(download_func, file_info): file_info for file_info in file_list}
            for future in concurrent.futures.as_completed(futures):
                if interrupted:
                    for f in futures:
                        f.cancel()
                    break
                completed += 1
                success, filename, status, size = future.result()
                if success:
                    if status == "downloaded":
                        successful += 1
                        bytes_downloaded += size
                        console.log(f"[green]✓ {filename} downloaded[/green]")
                    elif status == "skipped":
                        skipped += 1
                        console.log(f"[yellow]↷ {filename} skipped[/yellow]")
                else:
                    failed += 1
                    console.log(f"[red]✗ {filename} {status}[/red]")
                progress.update(status_task, completed=completed)
    if interrupted:
        console.log(f"[red]Interrupted. Completed: {successful}/{len(file_list)} files.[/red]")
    else:
        console.log(
            f"[green]Complete! Downloaded: {successful}, Skipped: {skipped}, Total size: {bytes_downloaded / (1024 ** 2):.1f} MB[/green]")


def main():
    valid_fragments = ['02110815', '03192025']

    # Set up command line argument parsing
    parser = argparse.ArgumentParser(description='Download scroll fragment layers')
    parser.add_argument('--fragment', '-f', type=str, help='Fragment ID to download')
    args = parser.parse_args()

    # If fragment is provided via command line
    if args.fragment:
        if args.fragment in valid_fragments:
            frag_id = args.fragment
            console.log(f"Using fragment ID from command line: {frag_id}")
        else:
            console.log(f"[red]Invalid fragment ID: {args.fragment}[/red]")
            console.log(f"Valid options are: {', '.join(valid_fragments)}")
            return
    else:
        # Interactive mode if no command line argument
        console.print("Select a fragment by entering the number (default: 1):")
        for idx, option in enumerate(valid_fragments, 1):
            console.print(f"\t{idx}. {option}")
        selection = console.input("Selection: ").strip()

        if selection == "2":
            frag_id = valid_fragments[1]
        else:
            if selection != "1" and selection:
                console.log("Invalid selection, defaulting to 1.")
            frag_id = valid_fragments[0]

    workers = os.cpu_count()

    url = f"https://dl.ash2txt.org/community-uploads/bruniss/scrolls/s5/autogens/{frag_id}/layers/"
    target_dir = f"data/scroll5/fragments/{frag_id}/layers"
    console.log(f"Downloading fragment: {frag_id}")
    download_files_parallel(url, target_dir, max_workers=workers)


if __name__ == "__main__":
    main()
