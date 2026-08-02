using System;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Security.Principal;

internal static class MaskPipelineDeployer
{
    private static string Quote(string value)
    {
        return "\"" + value.Replace("\"", "\\\"") + "\"";
    }

    private static bool IsAdministrator()
    {
        using (WindowsIdentity identity = WindowsIdentity.GetCurrent())
        {
            var principal = new WindowsPrincipal(identity);
            return principal.IsInRole(WindowsBuiltInRole.Administrator);
        }
    }

    public static int Main(string[] args)
    {
        try
        {
            string root = AppDomain.CurrentDomain.BaseDirectory;
            string script = Path.Combine(root, "Deploy-MaskPipeline.ps1");
            if (!File.Exists(script))
            {
                Console.Error.WriteLine("Deployment script was not found: " + script);
                return 2;
            }
            if (!IsAdministrator())
            {
                string executable = Process.GetCurrentProcess().MainModule.FileName;
                var elevate = new ProcessStartInfo(executable)
                {
                    UseShellExecute = true,
                    Verb = "runas",
                    Arguments = String.Join(" ", args.Select(Quote))
                };
                using (Process child = Process.Start(elevate))
                {
                    child.WaitForExit();
                    return child.ExitCode;
                }
            }
            string forwarded = String.Join(" ", args.Select(Quote));
            var start = new ProcessStartInfo("powershell.exe")
            {
                UseShellExecute = false,
                Arguments = "-NoProfile -ExecutionPolicy Bypass -File " + Quote(script) + " " + forwarded
            };
            using (Process process = Process.Start(start))
            {
                process.WaitForExit();
                Console.WriteLine(process.ExitCode == 0
                    ? "Mask Pipeline deployment completed successfully."
                    : "Mask Pipeline deployment failed. Review the deployment log.");
                Console.WriteLine("Press Enter to close.");
                Console.ReadLine();
                return process.ExitCode;
            }
        }
        catch (Exception error)
        {
            Console.Error.WriteLine(error);
            Console.WriteLine("Press Enter to close.");
            Console.ReadLine();
            return 1;
        }
    }
}
