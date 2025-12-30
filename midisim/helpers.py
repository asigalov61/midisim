#! /usr/bin/python3

r'''###############################################################################
###################################################################################
#
#	Helpers Python Module
#	Version 1.0
#
#	Project Los Angeles
#
#	Tegridy Code 2025
#
#   https://github.com/Tegridy-Code/Project-Los-Angeles
#
###################################################################################
###################################################################################
#
#   Copyright 2025 Project Los Angeles / Tegridy Code
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.
#
###################################################################################
'''

print('=' * 70)
print('Loading midisim helpers module...')
print('Please wait...')
print('=' * 70)

__version__ = '1.0.0'

print('midisim helpers module version', __version__)
print('=' * 70)

###################################################################################

import os
import shutil
import subprocess
import time

import importlib.resources as pkg_resources

from . import models
from . import embeddings

from typing import List, Dict

###################################################################################

def get_package_models() -> List[Dict]:
    
    """
    Get models included with midisim package
    
    Returns
    -------
    List of dicts: {'model': model_file_name,
                    'path': model_full_path
                   }
    """
    
    models_dict = []
    
    for resource in pkg_resources.contents(models):
        if resource.endswith('.pth'):
            with pkg_resources.path(models, resource) as p:
                mdic = {'model': resource,
                        'path': str(p)
                       }
                
                models_dict.append(mdic)
                
    return sorted(models_dict, key=lambda x: x['model'])

###################################################################################

def get_package_embeddings() -> List[Dict]:
    
    """
    Get pre-computed embeddings included with midisim package
    
    Returns
    -------
    List of dicts: {'embeddings': embeddings_file_name,
                    'path': embeddings_full_path
                   }
    """
    
    embeddings_dict = []
    
    for resource in pkg_resources.contents(embeddings):
        if resource.endswith('.npy'):
            with pkg_resources.path(embeddings, resource) as p:
                mdic = {'embeddings': resource,
                        'path': str(p)
                       }
                
                embeddings_dict.append(mdic)
                
    return sorted(embeddings_dict, key=lambda x: x['embeddings'])

###################################################################################

def is_installed(pkg: str) -> bool:
    """Return True if package is already installed (dpkg-query)."""
    try:
        subprocess.run(
            ["dpkg-query", "-W", "-f=${Status}", pkg],
            check=True, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True
        )
        # dpkg-query returns "install ok installed" on success
        out = subprocess.run(["dpkg-query", "-W", "-f=${Status}", pkg],
                             stdout=subprocess.PIPE, text=True).stdout.strip()
        return "installed" in out
    except subprocess.CalledProcessError:
        return False

###################################################################################

def _run_apt_get(args, timeout):
    base = ["apt-get", "-y", "-o", "Dpkg::Options::=--force-confdef", "-o", "Dpkg::Options::=--force-confold"]
    cmd = base + args
    return subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=timeout)

###################################################################################

def install_apt_package(pkg: str,
                        update: bool = True,
                        timeout: int = 600,
                        require_root: bool = True,
                        use_python_apt: bool = False
                        ) -> Dict:
    
    """
    Install an apt package idempotently.
    - pkg: package name (e.g., 'fluidsynth')
    - update: run apt-get update first
    - timeout: seconds for apt operations
    - require_root: if True, will prefix with sudo when not root (may prompt)
    - use_python_apt: try python-apt API first if True
    
    Returns
    -------
    Status dict {'status', 'package'}
    """
    
    if is_installed(pkg):
        return {"status": "already_installed", "package": pkg}

    # Optionally try python-apt (requires python-apt installed and running as root)
    if use_python_apt:
        try:
            import apt
            cache = apt.Cache()
            cache.update()
            cache.open(None)
            if pkg in cache:
                pkg_obj = cache[pkg]
                if not pkg_obj.is_installed:
                    pkg_obj.mark_install()
                    cache.commit()
                return {"status": "installed_via_python_apt", "package": pkg}
        except Exception:
            # fall through to subprocess fallback
            pass

    # Build command environment
    prefix = []
    if require_root and os.geteuid() != 0:
        if shutil.which("sudo"):
            prefix = ["sudo"]
        else:
            raise PermissionError("Root privileges required and sudo not available.")

    # Optionally update
    if update:
        tries = 5
        for attempt in range(tries):
            try:
                subprocess.run(prefix + ["apt-get", "update"], check=True, timeout=timeout, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                break
            except subprocess.CalledProcessError as e:
                if attempt + 1 == tries:
                    raise
                time.sleep(2 ** attempt)

    # Install with retry for transient locks
    tries = 6
    for attempt in range(tries):
        try:
            subprocess.run(prefix + ["apt-get", "-y", "install", pkg], check=True, timeout=timeout, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            if is_installed(pkg):
                return {"status": "installed", "package": pkg}
            else:
                raise RuntimeError("apt-get reported success but package not found installed.")
        except subprocess.CalledProcessError as e:
            # common cause: dpkg lock; backoff and retry
            if "Could not get lock" in e.stderr or "dpkg was interrupted" in e.stderr:
                time.sleep(2 ** attempt)
                continue
            raise
            
    raise RuntimeError("Failed to install package after retries.")

###################################################################################

print('Module is loaded!')
print('Enjoy! :)')
print('=' * 70)

###################################################################################
# This is the end of the Helpers Python Module
###################################################################################