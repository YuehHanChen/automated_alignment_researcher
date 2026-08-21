"""
S3 utilities for uploading and downloading ideas and results.

Includes robust handling for RunPod S3's quirks:
- Pagination bug workaround (level-by-level directory walking)
- Large file upload with timeout retry and HeadObject verification
- 524 error handling with exponential backoff
"""
import os
import json
import boto3
import uuid
import tarfile
import tempfile
import time
import math
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from threading import Lock
from botocore.exceptions import ClientError, BotoCoreError, ReadTimeoutError, ConnectTimeoutError
from botocore.config import Config


def get_s3_client():
    """Get S3 client using credentials from aar.config."""
    from aar.config import AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, S3_REGION, S3_ENDPOINT_URL

    if not AWS_ACCESS_KEY_ID or not AWS_SECRET_ACCESS_KEY:
        return None

    config = Config(
        read_timeout=7200,
        connect_timeout=60,
        retries={'max_attempts': 5, 'mode': 'adaptive'},
        max_pool_connections=50,
        signature_version='s3v4',
    )

    client_kwargs = {
        'aws_access_key_id': AWS_ACCESS_KEY_ID,
        'aws_secret_access_key': AWS_SECRET_ACCESS_KEY,
        'region_name': S3_REGION,
        'config': config,
    }

    if S3_ENDPOINT_URL:
        client_kwargs['endpoint_url'] = S3_ENDPOINT_URL

    return boto3.client('s3', **client_kwargs)


def download_results(
    idea_id: str,
    bucket_name: str,
    prefix: str = "results/",
) -> Optional[Dict[str, Any]]:
    """
    Download experiment results from S3.
    
    Args:
        idea_id: Unique identifier for the idea/experiment
        bucket_name: S3 bucket name
        prefix: S3 key prefix (default: "results/")
        
    Returns:
        Results dictionary, or None if not found
    """
    s3_client = get_s3_client()
    s3_key = f"{prefix}{idea_id}_results.json"
    
    try:
        response = s3_client.get_object(Bucket=bucket_name, Key=s3_key)
        results = json.loads(response['Body'].read().decode('utf-8'))
        return results
    except ClientError as e:
        if e.response['Error']['Code'] == 'NoSuchKey':
            return None
        raise Exception(f"Failed to download results from S3: {e}")


def generate_idea_uid() -> str:
    """
    Generate a unique identifier for an idea.
    
    Returns:
        UID string (e.g., "550e8400-e29b-41d4-a716-446655440000")
    """
    return str(uuid.uuid4())


def ensure_idea_has_uid(idea: Dict[str, Any]) -> str:
    """
    Ensure an idea has a UID, generating one if missing.
    
    Args:
        idea: Idea dictionary
        
    Returns:
        UID string
    """
    if "uid" not in idea:
        idea["uid"] = generate_idea_uid()
    return idea["uid"]


def idea_exists_in_s3(
    uid: str,
    bucket_name: str,
    prefix: str = "ideas/",
) -> bool:
    """
    Check if an idea already exists in S3.
    
    Args:
        uid: Idea UID
        bucket_name: S3 bucket name
        prefix: S3 key prefix (default: "ideas/")
        
    Returns:
        True if idea exists, False otherwise
    """
    s3_client = get_s3_client()
    idea_key = f"{prefix}{uid}/idea.json"
    
    try:
        s3_client.head_object(Bucket=bucket_name, Key=idea_key)
        return True
    except ClientError as e:
        if e.response['Error']['Code'] == '404' or e.response['Error']['Code'] == 'NoSuchKey':
            return False
        raise


def upload_idea_by_uid(
    idea: Dict[str, Any],
    bucket_name: str,
    prefix: str = "ideas/",
    metadata: Optional[Dict[str, Any]] = None,
    force_upload: bool = False,
) -> str:
    """
    Upload a single idea to S3 organized by UID.
    
    Args:
        idea: Idea dictionary (will get UID if missing)
        bucket_name: S3 bucket name
        prefix: S3 key prefix (default: "ideas/")
        metadata: Optional metadata to include
        force_upload: If True, upload even if already exists (default: False)
        
    Returns:
        UID of the idea
    """
    from datetime import datetime, timezone
    
    s3_client = get_s3_client()
    
    # Ensure idea has UID
    uid = ensure_idea_has_uid(idea)
    
    # Check if already exists
    if not force_upload and idea_exists_in_s3(uid, bucket_name, prefix):
        print(f"✓ Idea {idea.get('Name', 'unknown')} already exists in S3 (skipping upload)")
        print(f"  Location: s3://{bucket_name}/{prefix}{uid}/idea.json")
        return uid
    
    # Upload idea to ideas/{uid}/idea.json
    idea_key = f"{prefix}{uid}/idea.json"
    s3_client.put_object(
        Bucket=bucket_name,
        Key=idea_key,
        Body=json.dumps(idea, indent=2),
        ContentType='application/json',
    )
    print(f"✓ Uploaded idea {idea.get('Name', 'unknown')} to s3://{bucket_name}/{idea_key}")
    
    # Upload metadata to ideas/{uid}/metadata.json if provided
    if metadata:
        metadata_data = {
            "uid": uid,
            "idea_name": idea.get("Name", "unknown"),
            "uploaded_at": datetime.now(timezone.utc).isoformat(),
            **(metadata or {}),
        }
        metadata_key = f"{prefix}{uid}/metadata.json"
        s3_client.put_object(
            Bucket=bucket_name,
            Key=metadata_key,
            Body=json.dumps(metadata_data, indent=2),
            ContentType='application/json',
        )
        print(f"✓ Uploaded metadata to s3://{bucket_name}/{metadata_key}")
    
    return uid



class LargeFileUploader:
    """
    Robust uploader for large files to RunPod S3.
    
    Based on RunPod's official upload_large_file.py script:
    https://github.com/runpod/runpod-s3-examples/blob/main/upload_large_file.py
    
    Key features:
    - 50MB chunks (RunPod recommended)
    - 524 error handling with exponential backoff
    - CompleteMultipartUpload timeout handling with HeadObject verification
    - Automatic retry on timeouts
    """
    
    def __init__(
        self,
        file_path: Path,
        bucket: str,
        key: str,
        part_size: int = 50 * 1024 * 1024,  # 50MB (RunPod recommended)
        max_retries: int = 5,
        max_workers: int = 4,
    ):
        self.file_path = file_path
        self.bucket = bucket
        self.key = key
        self.part_size = part_size
        self.max_retries = max_retries
        self.max_workers = max_workers
        
        self.progress_lock = Lock()
        self.parts_completed = 0
        
        # Get S3 client with standard config
        self.s3 = get_s3_client()
        self.upload_id: Optional[str] = None
    
    @staticmethod
    def is_524_error(exc: Exception) -> bool:
        """Return True if the exception wraps a 524 timeout response."""
        if isinstance(exc, ClientError):
            meta = exc.response.get("ResponseMetadata", {})
            return meta.get("HTTPStatusCode") == 524
        return False
    
    @staticmethod
    def is_no_such_upload_error(exc: Exception) -> bool:
        """Return True if the exception reports a missing multipart upload."""
        if isinstance(exc, ClientError):
            err = exc.response.get("Error", {})
            return err.get("Code") == "NoSuchUpload"
        return False
    
    def call_with_retry(self, description: str, func, max_retries: Optional[int] = None):
        """Call func with retry on 524 or timeout errors."""
        retries = max_retries or self.max_retries
        for attempt in range(1, retries + 1):
            try:
                return func()
            except ClientError as exc:
                if self.is_524_error(exc):
                    print(f"   {description}: received 524 response (attempt {attempt})", flush=True)
                    if attempt == retries:
                        raise
                    backoff = 2 ** attempt
                    print(f"   {description}: retrying in {backoff}s...", flush=True)
                    time.sleep(backoff)
                    continue
                raise
            except (ReadTimeoutError, ConnectTimeoutError) as exc:
                print(f"   {description}: timeout (attempt {attempt}): {exc}", flush=True)
                if attempt == retries:
                    raise
                backoff = 2 ** attempt
                print(f"   {description}: retrying in {backoff}s...", flush=True)
                time.sleep(backoff)
        return None
    
    def upload_part(
        self,
        part_number: int,
        offset: int,
        bytes_to_read: int,
        total_parts: int,
        start_time: float,
    ) -> dict:
        """Upload a single part with exponential-backoff retries."""
        for attempt in range(1, self.max_retries + 1):
            try:
                with open(self.file_path, "rb") as f:
                    f.seek(offset)
                    data = f.read(bytes_to_read)
                
                resp = self.s3.upload_part(
                    Bucket=self.bucket,
                    Key=self.key,
                    PartNumber=part_number,
                    UploadId=self.upload_id,
                    Body=data,
                )
                etag = resp["ETag"]
                
                with self.progress_lock:
                    self.parts_completed += 1
                    progress = 100.0 * self.parts_completed / total_parts
                
                elapsed = time.time() - start_time
                if self.parts_completed > 0:
                    remaining = max(0, elapsed * (total_parts / self.parts_completed - 1))
                    eta = time.strftime("%Hh %Mm %Ss", time.gmtime(remaining))
                else:
                    eta = "?"
                
                if self.parts_completed <= 5 or self.parts_completed % 50 == 0:
                    print(f"   Part {part_number}/{total_parts}: uploaded, progress: {progress:.1f}%, ETA: {eta}", flush=True)
                
                return {"PartNumber": part_number, "ETag": etag}
                
            except (BotoCoreError, ClientError) as exc:
                if self.is_524_error(exc):
                    print(f"   Part {part_number}: 524 error (attempt {attempt})", flush=True)
                else:
                    print(f"   Part {part_number}: failed (attempt {attempt}): {exc}", flush=True)
                
                if attempt == self.max_retries:
                    raise
                
                backoff = 2 ** attempt
                time.sleep(backoff)
        
        raise RuntimeError(f"Failed to upload part {part_number}")
    
    def complete_with_verification(
        self,
        parts_sorted: list,
        initial_timeout: int,
        expected_size: int,
    ):
        """
        Complete the multipart upload with timeout handling.
        
        If CompleteMultipartUpload times out, wait and verify with HeadObject
        to check if the upload actually completed on the server side.
        """
        timeout = initial_timeout
        last_exc: Optional[Exception] = None
        
        for attempt in range(1, self.max_retries + 1):
            # Create client with current timeout
            from aar.config import AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, S3_REGION, S3_ENDPOINT_URL
            config = Config(
                read_timeout=timeout,
                connect_timeout=timeout,
                retries={'max_attempts': 3, 'mode': 'standard'},
            )
            client_kwargs = {
                'aws_access_key_id': AWS_ACCESS_KEY_ID,
                'aws_secret_access_key': AWS_SECRET_ACCESS_KEY,
                'region_name': S3_REGION,
                'config': config,
            }
            if S3_ENDPOINT_URL:
                client_kwargs['endpoint_url'] = S3_ENDPOINT_URL
            client = boto3.client('s3', **client_kwargs)
            
            try:
                client.complete_multipart_upload(
                    Bucket=self.bucket,
                    Key=self.key,
                    UploadId=self.upload_id,
                    MultipartUpload={"Parts": parts_sorted},
                )
                print(f"   ✓ CompleteMultipartUpload succeeded", flush=True)
                self.s3 = client
                return
                
            except (ReadTimeoutError, ConnectTimeoutError) as exc:
                last_exc = exc
                print(f"   ⚠️  CompleteMultipartUpload timed out after {timeout}s (attempt {attempt})", flush=True)
                no_such_upload = False
                
            except (ClientError, BotoCoreError) as exc:
                last_exc = exc
                no_such_upload = self.is_no_such_upload_error(exc)
                print(f"   ⚠️  CompleteMultipartUpload failed (attempt {attempt}): {exc}", flush=True)
            
            # Wait and verify with HeadObject
            if no_such_upload:
                print(f"   Upload session missing; checking object state immediately", flush=True)
                wait_time = 5
            else:
                wait_time = timeout
                print(f"   Waiting {wait_time}s before checking if upload completed on server...", flush=True)
            
            time.sleep(wait_time)
            
            try:
                head = self.call_with_retry(
                    "HeadObject",
                    lambda: client.head_object(Bucket=self.bucket, Key=self.key),
                )
                if head:
                    uploaded_size = head.get("ContentLength")
                    if uploaded_size == expected_size:
                        print(f"   ✓ HeadObject confirms upload completed (size: {uploaded_size} bytes)", flush=True)
                        self.s3 = client
                        return
                    print(f"   HeadObject size mismatch: {uploaded_size} vs expected {expected_size}", flush=True)
            except Exception as head_exc:
                print(f"   HeadObject failed: {head_exc}", flush=True)
            
            if attempt == self.max_retries:
                raise last_exc or RuntimeError("Exceeded max_retries for CompleteMultipartUpload")
            
            # Double timeout for next attempt
            timeout *= 2
            print(f"   Increasing timeout to {timeout}s for next attempt", flush=True)
    
    def upload(self) -> bool:
        """Execute the multipart upload."""
        file_size = self.file_path.stat().st_size
        total_parts = math.ceil(file_size / self.part_size)
        file_size_gb = file_size / (1024 ** 3)
        
        print(f"   File size: {file_size_gb:.2f} GB, will upload in {total_parts} parts of {self.part_size // (1024*1024)}MB each", flush=True)
        
        start_time = time.time()
        
        # Calculate completion timeout based on file size (5s per GB, min 60s)
        completion_timeout = max(120, int(math.ceil(file_size_gb) * 10))
        
        # Create multipart upload
        resp = self.call_with_retry(
            "CreateMultipartUpload",
            lambda: self.s3.create_multipart_upload(Bucket=self.bucket, Key=self.key),
        )
        self.upload_id = resp["UploadId"]
        print(f"   Created multipart upload: {self.upload_id[:20]}...", flush=True)
        
        parts: List[dict] = []
        try:
            # Upload parts in parallel
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                futures = {}
                for part_num in range(1, total_parts + 1):
                    offset = (part_num - 1) * self.part_size
                    chunk_size = min(self.part_size, file_size - offset)
                    futures[
                        executor.submit(
                            self.upload_part,
                            part_number=part_num,
                            offset=offset,
                            bytes_to_read=chunk_size,
                            total_parts=total_parts,
                            start_time=start_time,
                        )
                    ] = part_num
                
                for fut in as_completed(futures):
                    part = fut.result()
                    parts.append(part)
            
            # Verify all parts uploaded
            def fetch_parts():
                paginator = self.s3.get_paginator("list_parts")
                found = []
                for page in paginator.paginate(
                    Bucket=self.bucket, Key=self.key, UploadId=self.upload_id
                ):
                    found.extend(page.get("Parts", []))
                return found
            
            seen = self.call_with_retry("ListParts", fetch_parts)
            print(f"   Verified {len(seen)} of {total_parts} parts uploaded", flush=True)
            
            if len(seen) != total_parts:
                raise RuntimeError(f"Expected {total_parts} parts but saw {len(seen)}")
            
            # Complete multipart upload with verification
            parts_sorted = sorted(parts, key=lambda x: x["PartNumber"])
            print(f"   Sending CompleteMultipartUpload request...", flush=True)
            self.complete_with_verification(
                parts_sorted=parts_sorted,
                initial_timeout=completion_timeout,
                expected_size=file_size,
            )
            
            # Final verification
            head = self.call_with_retry(
                "HeadObject (final)",
                lambda: self.s3.head_object(Bucket=self.bucket, Key=self.key),
            )
            if head:
                uploaded_size = head.get("ContentLength")
                if uploaded_size != file_size:
                    raise RuntimeError(f"Size mismatch: remote {uploaded_size} vs local {file_size}")
                print(f"   ✓ Verified upload: {uploaded_size} bytes", flush=True)
            
            elapsed = time.time() - start_time
            speed = (file_size / (1024 * 1024)) / elapsed if elapsed > 0 else 0
            duration = time.strftime("%Hh %Mm %Ss", time.gmtime(elapsed))
            print(f"   Upload complete: {speed:.2f} MB/s, duration: {duration}", flush=True)
            
            return True
            
        except Exception as exc:
            print(f"   ❌ Upload failed: {exc}", flush=True)
            if self.upload_id:
                print(f"   UploadId {self.upload_id} left open for potential resumption", flush=True)
            raise


def upload_directory_to_s3(
    directory_path: Path,
    s3_key_prefix: str,
    bucket_name: str,
    exclude_patterns: list = None,
) -> str:
    """
    Upload a directory to S3 as a compressed tar.gz archive.
    
    Uses robust multipart upload with:
    - 50MB chunks (RunPod recommended)
    - 524 error handling with exponential backoff  
    - CompleteMultipartUpload timeout handling with HeadObject verification
    
    Args:
        directory_path: Path to directory to upload
        s3_key_prefix: S3 key prefix (e.g., "ideas/{uid}/workspace.tar.gz")
        bucket_name: S3 bucket name
        exclude_patterns: List of directory/file names to exclude (e.g., ["results", ".git", "__pycache__"])
        
    Returns:
        S3 key where the archive was uploaded
    """
    if not directory_path.exists():
        raise FileNotFoundError(f"Directory not found: {directory_path}")
    
    if not directory_path.is_dir():
        raise ValueError(f"Path is not a directory: {directory_path}")
    
    # Create temporary tar.gz file
    with tempfile.NamedTemporaryFile(suffix='.tar.gz', delete=False) as tmp_file:
        tmp_path = Path(tmp_file.name)
    
    # Default exclude patterns (always exclude these to reduce archive size)
    default_excludes = {'.git', '__pycache__', '.pytest_cache', 'node_modules', '.venv', 'venv', 'results', 'unsloth_compiled_cache'}
    exclude_set = default_excludes.copy()
    if exclude_patterns:
        exclude_set.update(exclude_patterns)
    
    try:
        # Create tar.gz archive
        print(f"📦 Creating archive of {directory_path}...", flush=True)
        if exclude_set:
            print(f"   Excluding: {', '.join(sorted(exclude_set))}", flush=True)
        
        with tarfile.open(tmp_path, 'w:gz') as tar:
            files_added = 0
            files_skipped = 0
            total_size = 0
            for root, dirs, files in os.walk(directory_path):
                # Filter out excluded directories (modifies dirs in-place to prevent os.walk from descending)
                dirs[:] = [d for d in dirs if d not in exclude_set]
                
                for file in files:
                    # Skip excluded files
                    if file in exclude_set:
                        files_skipped += 1
                        continue
                    
                    file_path = Path(root) / file
                    # Use relative path from directory_path
                    arcname = file_path.relative_to(directory_path.parent)
                    tar.add(file_path, arcname=arcname)
                    files_added += 1
                    total_size += file_path.stat().st_size
            
            print(f"   Added {files_added} files ({total_size / (1024*1024):.2f} MB) to archive", flush=True)
            if files_skipped > 0:
                print(f"   Skipped {files_skipped} excluded files", flush=True)
        
        # Get file size before upload
        file_size = tmp_path.stat().st_size
        file_size_mb = file_size / (1024 * 1024)
        file_size_gb = file_size / (1024 * 1024 * 1024)
        
        print(f"📤 Uploading archive ({file_size_mb:.2f} MB) to s3://{bucket_name}/{s3_key_prefix}...", flush=True)
        
        # For large files (>1GB), use robust multipart uploader
        if file_size_gb > 1:
            print(f"   Using robust multipart uploader for large file ({file_size_gb:.1f} GB)", flush=True)
            uploader = LargeFileUploader(
                file_path=tmp_path,
                bucket=bucket_name,
                key=s3_key_prefix,
                part_size=50 * 1024 * 1024,  # 50MB chunks (RunPod recommended)
                max_retries=5,
                max_workers=4,
            )
            uploader.upload()
        else:
            # For smaller files, use standard boto3 upload
            from boto3.s3.transfer import TransferConfig
            
            s3_client = get_s3_client()
            transfer_config = TransferConfig(
                multipart_threshold=1024 * 1024 * 8,  # 8 MB
                multipart_chunksize=1024 * 1024 * 50,  # 50 MB
                max_concurrency=4,
                use_threads=True,
            )
            
            s3_client.upload_file(
                Filename=str(tmp_path),
                Bucket=bucket_name,
                Key=s3_key_prefix,
                ExtraArgs={'ContentType': 'application/gzip'},
                Config=transfer_config,
            )
        
        print(f"✓ Uploaded workspace archive ({file_size_mb:.2f} MB) to s3://{bucket_name}/{s3_key_prefix}", flush=True)
        return s3_key_prefix
        
    finally:
        # Clean up temporary file
        if tmp_path.exists():
            tmp_path.unlink()


def upload_file_to_s3(
    file_path: Path,
    s3_key: str,
    bucket_name: str,
    content_type: str = "text/plain",
) -> str:
    """
    Upload a single file to S3.
    
    Args:
        file_path: Path to file to upload
        s3_key: S3 key (full path)
        bucket_name: S3 bucket name
        content_type: Content type for the file (default: "text/plain")
        
    Returns:
        S3 key where the file was uploaded
    """
    s3_client = get_s3_client()
    
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    print(f"📤 Uploading {file_path.name} to s3://{bucket_name}/{s3_key}...")
    
    # Use boto3's upload_file() which automatically handles multipart uploads for large files
    s3_client.upload_file(
        Filename=str(file_path),
        Bucket=bucket_name,
        Key=s3_key,
        ExtraArgs={'ContentType': content_type},
    )
    
    file_size_mb = file_path.stat().st_size / (1024 * 1024)
    print(f"✓ Uploaded {file_path.name} ({file_size_mb:.2f} MB) to s3://{bucket_name}/{s3_key}")
    
    return s3_key


def _list_s3_prefixes(s3_client, bucket_name: str, prefix: str) -> List[str]:
    """
    List immediate subdirectory prefixes under an S3 prefix.
    Uses Delimiter='/' to get only immediate children.
    
    Returns list of prefixes (directories) found.
    """
    prefixes = []
    try:
        response = s3_client.list_objects_v2(
            Bucket=bucket_name,
            Prefix=prefix,
            Delimiter='/',
            MaxKeys=1000
        )
        if 'CommonPrefixes' in response:
            prefixes = [p['Prefix'] for p in response['CommonPrefixes']]
    except Exception as e:
        print(f"[DEBUG] Error listing prefixes for {prefix}: {e}", flush=True)
    return prefixes


def _list_s3_files(s3_client, bucket_name: str, prefix: str) -> List[Dict[str, Any]]:
    """
    List files (not directories) directly under an S3 prefix.
    Uses Delimiter='/' to get only immediate children.
    
    Returns list of object dicts with 'Key' and 'Size'.
    """
    files = []
    try:
        response = s3_client.list_objects_v2(
            Bucket=bucket_name,
            Prefix=prefix,
            Delimiter='/',
            MaxKeys=1000
        )
        if 'Contents' in response:
            # Filter out the prefix itself (directory marker)
            files = [obj for obj in response['Contents'] 
                     if obj['Key'] != prefix and not obj['Key'].endswith('/')]
    except Exception as e:
        print(f"[DEBUG] Error listing files for {prefix}: {e}", flush=True)
    return files


def _walk_s3_directory(s3_client, bucket_name: str, prefix: str) -> List[Dict[str, Any]]:
    """
    Recursively walk an S3 directory structure, level by level.
    
    This is a workaround for RunPod S3's pagination bug where recursive
    listing returns IsTruncated=True with empty pages.
    
    Returns list of all file objects found.
    """
    all_files = []
    prefixes_to_visit = [prefix]
    visited_prefixes = set()
    
    while prefixes_to_visit:
        current_prefix = prefixes_to_visit.pop(0)
        
        # Skip if already visited (avoid infinite loops)
        if current_prefix in visited_prefixes:
            continue
        visited_prefixes.add(current_prefix)
        
        # Get files at this level
        files = _list_s3_files(s3_client, bucket_name, current_prefix)
        all_files.extend(files)
        
        # Get subdirectories and add to queue
        sub_prefixes = _list_s3_prefixes(s3_client, bucket_name, current_prefix)
        for sub_prefix in sub_prefixes:
            if sub_prefix not in visited_prefixes:
                prefixes_to_visit.append(sub_prefix)
    
    return all_files


def download_s3_directory(
    local_dir: Path,
    bucket_name: str,
    s3_prefix: str,
    force_download: bool = False,
    description: str = "files",
) -> bool:
    """
    Download a directory from S3 if it doesn't exist locally.
    
    Downloads all files under the s3_prefix to the local_dir,
    preserving the directory structure.
    
    Uses level-by-level directory walking to work around RunPod S3's
    pagination bug with recursive listing.
    
    Args:
        local_dir: Local directory where files should be stored
        bucket_name: S3 bucket name
        s3_prefix: S3 prefix (e.g., "cache_results/", "results/", "data/")
        force_download: If True, download even if local files exist (default: False)
        description: Human-readable description for logging (e.g., "cache", "results")
        
    Returns:
        True if files were downloaded or already exist, False if download failed
    """
    s3_client = get_s3_client()
    
    # Ensure local directory exists
    local_dir.mkdir(parents=True, exist_ok=True)

    # Check existing local files (for logging purposes)
    existing_files = [f for f in local_dir.rglob("*") if f.is_file()]
    if existing_files:
        print(f"ℹ️  Found {len(existing_files)} existing files in {local_dir}", flush=True)

    print(f"\n{'='*80}", flush=True)
    print(f"Downloading {description} from S3", flush=True)
    print(f"{'='*80}", flush=True)
    print(f"Local directory: {local_dir}", flush=True)
    print(f"S3 path: s3://{bucket_name}/{s3_prefix}", flush=True)
    print(f"{'='*80}\n", flush=True)
    
    try:
        print(f"[DEBUG] Walking S3 directory structure...", flush=True)
        
        # Use level-by-level walking instead of recursive pagination
        all_files = _walk_s3_directory(s3_client, bucket_name, s3_prefix)
        
        print(f"[DEBUG] Found {len(all_files)} files to download", flush=True)
        
        # Filter files to download
        files_to_download = []
        for obj in all_files:
            s3_key = obj['Key']
            
            # Skip if it's just the prefix directory marker
            if s3_key == s3_prefix or s3_key.endswith('/'):
                continue
            
            # Skip if key doesn't start with our exact prefix (safety check)
            if not s3_key.startswith(s3_prefix):
                continue
            
            # Calculate local path (remove s3_prefix from key)
            relative_path = s3_key[len(s3_prefix):]
            if not relative_path:
                continue
                
            local_path = local_dir / relative_path
            
            # Skip if file already exists locally (unless force_download)
            if local_path.exists() and not force_download:
                continue
            
            files_to_download.append((s3_key, relative_path, local_path))
        
        print(f"[DEBUG] {len(files_to_download)} files need downloading", flush=True)
        
        if not files_to_download:
            print(f"[DEBUG] No files to download", flush=True)
            return True
        
        # Download multiple files in parallel using ThreadPoolExecutor
        import concurrent.futures
        
        downloaded_count = 0
        total_size = 0
        lock = threading.Lock()
        
        def download_one(args):
            nonlocal downloaded_count, total_size
            s3_key, relative_path, local_path = args
            
            try:
                local_path.parent.mkdir(parents=True, exist_ok=True)
                s3_client.download_file(bucket_name, s3_key, str(local_path))
                
                file_size = local_path.stat().st_size
                with lock:
                    downloaded_count += 1
                    total_size += file_size
                    if downloaded_count <= 5 or downloaded_count % 50 == 0:
                        print(f"  ✓ {downloaded_count}/{len(files_to_download)}: {relative_path}", flush=True)
                return True
            except Exception as e:
                print(f"  ⚠️ Failed: {relative_path}: {e}", flush=True)
                return False
        
        max_workers = min(20, len(files_to_download))
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            executor.map(download_one, files_to_download)
        
        print(f"[DEBUG] Finished downloading, downloaded_count={downloaded_count}", flush=True)
        
        if downloaded_count > 0:
            total_size_mb = total_size / (1024 * 1024)
            print(f"\n✓ Downloaded {downloaded_count} {description} ({total_size_mb:.2f} MB) to {local_dir}", flush=True)
            return True
        else:
            print(f"\n⚠️  No {description} found in S3 at s3://{bucket_name}/{s3_prefix}", flush=True)
            return False
            
    except ClientError as e:
        error_code = e.response.get('Error', {}).get('Code', '')
        if error_code in ('NoSuchBucket', '404'):
            print(f"⚠️  S3 bucket {bucket_name} not found", flush=True)
        else:
            print(f"⚠️  Failed to list {description} from S3: {e}", flush=True)
        return False
    except Exception as e:
        print(f"⚠️  Error downloading {description} from S3: {e}", flush=True)
        return False


# =============================================================================
# Commit/Checkpoint Functions
# =============================================================================


def generate_commit_id(
    experiment_id: int,
    sequence_number: int,
    message: str,
    timestamp: str,
) -> str:
    """
    Generate a unique commit ID using SHA256 hash.

    Args:
        experiment_id: Database experiment ID
        sequence_number: Sequence number within the run (0, 1, 2, ...)
        message: Commit message
        timestamp: ISO format timestamp

    Returns:
        16-character hex string (SHA256 prefix)
    """
    import hashlib

    content = json.dumps({
        'experiment_id': experiment_id,
        'sequence_number': sequence_number,
        'message': message,
        'timestamp': timestamp,
    }, sort_keys=True)
    return hashlib.sha256(content.encode()).hexdigest()[:16]


def upload_commit_to_s3(
    idea_uid: str,
    run_id: str,
    commit_id: str,
    workspace_dir: Path,
    metadata: Dict[str, Any],
    bucket_name: str,
    prefix: str = "ideas/",
    exclude_patterns: Optional[List[str]] = None,
) -> Tuple[str, int, List[str]]:
    """
    Upload a commit (checkpoint) to S3.

    Creates:
    - ideas/{idea_uid}/{run_id}/commits/{commit_id}/metadata.json
    - ideas/{idea_uid}/{run_id}/commits/{commit_id}/workspace.tar.gz

    Uses same include/exclude patterns as upload_directory_to_s3 (uploads ALL files
    except excluded directories).

    Args:
        idea_uid: Idea UID
        run_id: Run ID
        commit_id: Commit ID (16-char hash)
        workspace_dir: Directory to archive
        metadata: Commit metadata (message, metrics, approach, etc.)
        bucket_name: S3 bucket name
        prefix: S3 key prefix (default: "ideas/")
        exclude_patterns: Additional patterns to exclude (merged with defaults)

    Returns:
        Tuple of (s3_key_prefix, archive_size_bytes, files_list)
    """
    from datetime import datetime, timezone
    from boto3.s3.transfer import TransferConfig

    s3_client = get_s3_client()

    # S3 key prefix for this commit
    s3_key_prefix = f"{prefix}{idea_uid}/{run_id}/commits/{commit_id}/"

    # Use same exclude patterns as upload_directory_to_s3 (upload ALL files except these)
    default_excludes = {'.git', '__pycache__', '.pytest_cache', 'node_modules', '.venv', 'venv', 'results', 'unsloth_compiled_cache'}
    exclude_set = default_excludes.copy()
    if exclude_patterns:
        exclude_set.update(exclude_patterns)

    # Collect ALL files (same as upload_directory_to_s3)
    files_to_include = []
    workspace_path = Path(workspace_dir)

    if workspace_path.exists() and workspace_path.is_dir():
        for root, dirs, files in os.walk(workspace_path):
            # Filter out excluded directories (modifies dirs in-place)
            dirs[:] = [d for d in dirs if d not in exclude_set]

            for file in files:
                # Skip excluded files
                if file in exclude_set:
                    continue

                file_path = Path(root) / file
                rel_path = str(file_path.relative_to(workspace_path))
                files_to_include.append(rel_path)

    files_to_include = sorted(files_to_include)

    # Upload metadata.json using put_object (small file)
    metadata_with_timestamp = {
        **metadata,
        'commit_id': commit_id,
        'idea_uid': idea_uid,
        'run_id': run_id,
        'files': files_to_include,
        'created_at': datetime.now(timezone.utc).isoformat(),
    }
    metadata_key = f"{s3_key_prefix}metadata.json"
    s3_client.put_object(
        Bucket=bucket_name,
        Key=metadata_key,
        Body=json.dumps(metadata_with_timestamp, indent=2),
        ContentType='application/json',
    )
    print(f"✓ Uploaded commit metadata to s3://{bucket_name}/{metadata_key}")

    # Create and upload workspace archive (same logic as upload_directory_to_s3)
    archive_size = 0
    if files_to_include and workspace_path.exists():
        with tempfile.NamedTemporaryFile(suffix='.tar.gz', delete=False) as tmp_file:
            tmp_path = Path(tmp_file.name)

        try:
            print(f"📦 Creating commit archive of {workspace_path}...", flush=True)
            print(f"   Excluding: {', '.join(sorted(exclude_set))}", flush=True)

            with tarfile.open(tmp_path, 'w:gz') as tar:
                for rel_path in files_to_include:
                    file_path = workspace_path / rel_path
                    if file_path.exists():
                        tar.add(file_path, arcname=rel_path)

            print(f"   Added {len(files_to_include)} files to archive", flush=True)

            archive_size = tmp_path.stat().st_size
            workspace_key = f"{s3_key_prefix}workspace.tar.gz"

            # Use robust upload with TransferConfig (same as upload_directory_to_s3)
            file_size_gb = archive_size / (1024 * 1024 * 1024)
            if file_size_gb > 1:
                # Large file: use LargeFileUploader
                print(f"   Using robust multipart uploader for large file ({file_size_gb:.1f} GB)")
                uploader = LargeFileUploader(
                    file_path=tmp_path,
                    bucket=bucket_name,
                    key=workspace_key,
                    part_size=50 * 1024 * 1024,
                    max_retries=5,
                    max_workers=4,
                )
                uploader.upload()
            else:
                # Small file: standard upload with TransferConfig
                transfer_config = TransferConfig(
                    multipart_threshold=1024 * 1024 * 8,
                    multipart_chunksize=1024 * 1024 * 50,
                    max_concurrency=4,
                    use_threads=True,
                )
                s3_client.upload_file(
                    Filename=str(tmp_path),
                    Bucket=bucket_name,
                    Key=workspace_key,
                    ExtraArgs={'ContentType': 'application/gzip'},
                    Config=transfer_config,
                )

            print(f"✓ Uploaded commit workspace ({archive_size / (1024*1024):.2f} MB) to s3://{bucket_name}/{workspace_key}")

        finally:
            if tmp_path.exists():
                tmp_path.unlink()

    return s3_key_prefix, archive_size, files_to_include


def download_snapshot_from_s3(
    commit_id: str,
    idea_uid: str,
    run_id: str,
    target_dir: Path,
    bucket_name: str,
    prefix: str = "ideas/",
) -> Tuple[Dict[str, Any], List[str]]:
    """
    Download a snapshot (commit/checkpoint) from S3.

    Downloads:
    - ideas/{idea_uid}/{run_id}/commits/{commit_id}/metadata.json
    - ideas/{idea_uid}/{run_id}/commits/{commit_id}/workspace.tar.gz (extracted)

    Args:
        commit_id: Commit ID (16-char hash)
        idea_uid: Idea UID
        run_id: Run ID
        target_dir: Directory to extract workspace to
        bucket_name: S3 bucket name
        prefix: S3 key prefix (default: "ideas/")

    Returns:
        Tuple of (metadata_dict, extracted_files_list)
    """
    s3_client = get_s3_client()

    # S3 key prefix for this commit
    s3_key_prefix = f"{prefix}{idea_uid}/{run_id}/commits/{commit_id}/"

    # Create target directory
    target_path = Path(target_dir)
    target_path.mkdir(parents=True, exist_ok=True)

    # Download metadata.json
    metadata_key = f"{s3_key_prefix}metadata.json"
    try:
        response = s3_client.get_object(Bucket=bucket_name, Key=metadata_key)
        metadata = json.loads(response['Body'].read().decode('utf-8'))
        print(f"✓ Downloaded commit metadata from s3://{bucket_name}/{metadata_key}")
    except ClientError as e:
        error_code = e.response.get('Error', {}).get('Code', '')
        if error_code in ('NoSuchKey', '404'):
            raise FileNotFoundError(f"Commit {commit_id} not found in S3")
        raise

    # Download and extract workspace.tar.gz
    workspace_key = f"{s3_key_prefix}workspace.tar.gz"
    extracted_files = []

    try:
        with tempfile.NamedTemporaryFile(suffix='.tar.gz', delete=False) as tmp_file:
            tmp_path = Path(tmp_file.name)

        # Use streaming download for better compatibility with RunPod S3
        download_success = download_s3_file_streaming(s3_client, bucket_name, workspace_key, str(tmp_path))
        if not download_success:
            # Fallback to standard download
            s3_client.download_file(bucket_name, workspace_key, str(tmp_path))

        print(f"✓ Downloaded commit workspace from s3://{bucket_name}/{workspace_key}")

        # Extract archive with security checks
        with tarfile.open(tmp_path, 'r:gz') as tar:
            for member in tar.getmembers():
                member_path = target_path / member.name
                # Security: prevent path traversal
                if not str(member_path.resolve()).startswith(str(target_path.resolve())):
                    print(f"⚠️ Skipping unsafe path: {member.name}")
                    continue
                extracted_files.append(member.name)

            tar.extractall(path=target_path)
            print(f"✓ Extracted {len(extracted_files)} files to {target_path}")

        tmp_path.unlink()

    except ClientError as e:
        error_code = e.response.get('Error', {}).get('Code', '')
        if error_code in ('NoSuchKey', '404'):
            print(f"ℹ️ No workspace archive found for commit {commit_id}")
        else:
            raise

    return metadata, extracted_files


def download_s3_file_streaming(s3_client, bucket: str, key: str, local_path: str, chunk_size: int = 8 * 1024 * 1024) -> bool:
    """
    Download S3 file using streaming for better compatibility with RunPod S3.

    Note: RunPod S3 may take several minutes for the first request to large files
    because it computes and caches MD5 checksums (ETags) on first access.
    """
    import time

    try:
        start_time = time.time()
        response = s3_client.get_object(Bucket=bucket, Key=key)
        body = response['Body']

        bytes_downloaded = 0
        with open(local_path, 'wb') as f:
            while True:
                chunk = body.read(chunk_size)
                if not chunk:
                    break
                f.write(chunk)
                bytes_downloaded += len(chunk)

        total_time = time.time() - start_time
        if total_time > 1:
            print(f"  Download: {bytes_downloaded / (1024*1024):.0f} MB in {total_time:.1f}s")
        return True

    except Exception as e:
        print(f"Streaming download failed: {e}")
        return False

