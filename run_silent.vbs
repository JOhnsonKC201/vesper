' Start Vesper with no console window. Put a shortcut to this in your Startup
' folder if you want it running from login.
Set shell = CreateObject("WScript.Shell")
shell.CurrentDirectory = CreateObject("Scripting.FileSystemObject").GetParentFolderName(WScript.ScriptFullName)
shell.Run "cmd /c run.bat", 0, False
