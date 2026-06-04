"""
Build script: PyInstaller -> .exe -> .msi via WiX Toolset.

Usage:
    python build_windows.py          # build .exe only
    python build_windows.py --msi    # build .exe + .msi (requires WiX installed)

Requirements:
    pip install pyinstaller
    WiX Toolset (for .msi): https://wixtoolset.org/
"""

import os
import sys
import shutil
import subprocess
import argparse
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
DIST_DIR = PROJECT_DIR / "dist"
BUILD_DIR = PROJECT_DIR / "build"
SPEC_FILE = PROJECT_DIR / "alpr.spec"
APP_NAME = "ALPRSystem"

PYINSTALLER_CMD = [
    sys.executable, "-m", "PyInstaller",
    "--name", APP_NAME,
    "--windowed",
    "--onefile",
    "--add-data", f"{PROJECT_DIR / 'data'}{os.pathsep}data",
    "--add-data", f"{PROJECT_DIR / 'models'}{os.pathsep}models",
    "--add-data", f"{PROJECT_DIR / 'ui'}{os.pathsep}ui",
    "--add-data", f"{PROJECT_DIR / 'processor'}{os.pathsep}processor",
    "--add-data", f"{PROJECT_DIR / 'utils'}{os.pathsep}utils",
    "--hidden-import", "ultralytics",
    "--hidden-import", "easyocr",
    "--hidden-import", "cv2",
    "--hidden-import", "sklearn.cluster",
    "--collect-all", "ultralytics",
    "--collect-all", "easyocr",
    "--distpath", str(DIST_DIR),
    "--workpath", str(BUILD_DIR),
    str(PROJECT_DIR / "main.py"),
]

MSI_CMD = [
    "candle", str(PROJECT_DIR / "installer" / "alpr.wxs"),
    "-out", str(DIST_DIR / "alpr.wixobj"),
]
LIGHT_CMD = [
    "light", str(DIST_DIR / "alpr.wixobj"),
    "-out", str(DIST_DIR / f"{APP_NAME}.msi"),
    "-ext", "WixUIExtension",
]


def build_exe():
    print("=" * 60)
    print("Building standalone .exe with PyInstaller...")
    print("=" * 60)

    if DIST_DIR.exists():
        shutil.rmtree(DIST_DIR)

    result = subprocess.run(PYINSTALLER_CMD)
    if result.returncode != 0:
        print("PyInstaller build FAILED.")
        sys.exit(1)

    print(f"\n.exe created at: {DIST_DIR / APP_NAME / f'{APP_NAME}.exe'}")
    return DIST_DIR / APP_NAME / f"{APP_NAME}.exe"


def build_msi(exe_path: Path):
    print("\n" + "=" * 60)
    print("Building .msi installer with WiX Toolset...")
    print("=" * 60)

    installer_dir = PROJECT_DIR / "installer"
    installer_dir.mkdir(exist_ok=True)

    wxs_path = installer_dir / "alpr.wxs"
    if not wxs_path.exists():
        _generate_wxs(wxs_path, exe_path)

    subprocess.run(MSI_CMD, check=True)
    subprocess.run(LIGHT_CMD, check=True)

    msi_path = DIST_DIR / f"{APP_NAME}.msi"
    print(f"\n.msi created at: {msi_path}")


def _generate_wxs(wxs_path: Path, exe_path: Path):
    content = f"""<?xml version="1.0" encoding="UTF-8"?>
<Wix xmlns="http://schemas.microsoft.com/wix/2006/wi">
    <Product Id="*" Name="{APP_NAME}" Language="1033"
             Version="1.0.0" Manufacturer="ALPR"
             UpgradeCode="A1B2C3D4-E5F6-7890-ABCD-EF1234567890">
        <Package InstallerVersion="200" Compressed="yes" />
        <Media Id="1" Cabinet="product.cab" EmbedCab="yes" />

        <Directory Id="TARGETDIR" Name="SourceDir">
            <Directory Id="ProgramFiles64Folder">
                <Directory Id="APPLICATIONFOLDER" Name="{APP_NAME}">
                    <Component Id="MainExecutable" Guid="*">
                        <File Id="ALPR_EXE" Name="{APP_NAME}.exe"
                              Source="{exe_path}" KeyPath="yes" />
                    </Component>
                    <Component Id="DataFolder" Guid="*">
                        <CreateFolder />
                        <Directory Id="DataDir" Name="data" />
                    </Component>
                    <Component Id="ModelsFolder" Guid="*">
                        <CreateFolder />
                        <Directory Id="ModelsDir" Name="models" />
                    </Component>
                </Directory>
            </Directory>
        </Directory>

        <Feature Id="MainFeature" Title="{APP_NAME}" Level="1">
            <ComponentRef Id="MainExecutable" />
            <ComponentRef Id="DataFolder" />
            <ComponentRef Id="ModelsFolder" />
        </Feature>

        <UI>
            <UIRef Id="WixUI_Minimal" />
        </UI>
    </Product>
</Wix>"""
    wxs_path.write_text(content, encoding="utf-8")
    print(f"Generated: {wxs_path}")


def main():
    parser = argparse.ArgumentParser(description="Build ALPR Windows installer")
    parser.add_argument("--msi", action="store_true", help="Also build .msi")
    args = parser.parse_args()

    exe_path = build_exe()
    if args.msi:
        build_msi(exe_path)


if __name__ == "__main__":
    main()
