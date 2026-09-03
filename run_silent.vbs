' Start Vesper with no console window. Put a shortcut to this in your Startup
' folder if you want it running from login.
'
' Anything passed to this script is handed straight to run.bat, which passes it
' on to vesper.main. The resume task in `vesper/resume.py` relies on that to send
' --if-idle, so an unlock while Vesper is already running exits quietly instead
' of putting a message box on screen.
Set shell = CreateObject("WScript.Shell")
shell.CurrentDirectory = CreateObject("Scripting.FileSystemObject").GetParentFolderName(WScript.ScriptFullName)
extra = ""
For Each arg In WScript.Arguments
  extra = extra & " " & arg
Next
shell.Run "cmd /c run.bat" & extra, 0, False
