@echo off
set /p msg="Enter commit message: "
if "%msg%"=="" set msg="Auto update"

echo Adding changes...
git add .

echo Committing...
git commit -m "%msg%"

echo Pushing to GitHub...
git push -u origin master

echo Done!
pause
