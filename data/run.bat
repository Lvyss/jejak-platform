@echo off
setlocal enabledelayedexpansion

set "count=1"

rem Mengambil semua file IMG_20261122_*.jpg dan mengurutkannya berdasarkan nama
for /f "delims=" %%f in ('dir /b /o:n "IMG_20261122_*.jpg"') do (
    
    rem Membuat nama folder (obj01, obj02, dst)
    set "folderName=obj0!count!"
    
    rem Jika hitungan sudah 10 ke atas, hilangkan angka 0 di depan
    if !count! GEQ 10 (
        set "folderName=obj!count!"
    )
    
    rem Proses pemindahan dan rename ke TG.jpg
    echo Memindahkan "%%f" ke !folderName!\TG.jpg
    move "%%f" "!folderName!\TG.jpg"
    
    set /a count+=1
)

echo.
echo =====================================
echo Selesai broo! Semua file sudah di-rename jadi TG.jpg
echo =====================================
pause