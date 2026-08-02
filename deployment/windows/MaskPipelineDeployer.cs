using System;
using System.Diagnostics;
using System.IO;
using System.Linq;

internal static class MaskPipelineDeployer
{
    private static string Quote(string value)
    {
        return "\"" + value.Replace("\"", "\\\"") + "\"";
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
