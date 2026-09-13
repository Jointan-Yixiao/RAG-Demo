// 资料问答工作台 · Windows 启动器
// 编译器：C:\Windows\Microsoft.NET\Framework64\v4.0.30319\csc.exe（C# 5 语法，无外部包）。
// 构建：scripts/_build_launcher.ps1。设计说明：doc/启动器实现说明.md。
//
// 结构：
//   LauncherOptions   命令行解析（纯函数，可测）
//   WinArgs           Windows 命令行参数转义（纯函数，可测）
//   Workspace         工作区身份 workspace_id（纯函数，可测）
//   HealthInfo        /api/health 解析与“是不是本项目的服务”判定（纯函数，可测）
//   CheckInfo/SetupOutput  安装后端 --check JSON 与 STAGE 行解析（纯函数，可测）
//   ShortcutManager   桌面快捷方式创建/修复/冲突处理（WScript.Shell COM）
//   OwnedJob / ProcessRunner  隐藏、异步子进程；只终止启动器自己创建的进程树
//   LauncherForm      WinForms 界面
//   Program           入口：--install-shortcut 走无界面 CLI，否则显示界面

using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Diagnostics;
using System.Drawing;
using System.Drawing.Drawing2D;
using System.IO;
using System.Net;
using System.Reflection;
using System.Runtime.InteropServices;
using System.Security.Cryptography;
using System.Text;
using System.Text.RegularExpressions;
using System.Threading;
using System.Web.Script.Serialization;
using System.Windows.Forms;

namespace RagWorkbench
{
    // ------------------------------------------------------------------ 常量

    public static class AppInfo
    {
        public const string Version = "1.0";
        public const int Port = 8765;
        public const string ExeName = "RAG-Workbench.exe";
        public const string ShortcutBaseName = "RAG 工作台";
        public const string HealthService = "rag-workbench";

        public static string BaseUrl { get { return "http://127.0.0.1:" + Port + "/"; } }
        public static string HealthUrl { get { return "http://127.0.0.1:" + Port + "/api/health"; } }
    }

    public static class ExitCodes
    {
        public const int Ok = 0;
        public const int Failure = 1;
        public const int ShortcutConflict = 2;
        public const int ShortcutFailed = 3;
        public const int AlreadyRunning = 4;
        public const int Usage = 64;
    }

    // ------------------------------------------------------------------ 命令行

    public sealed class LauncherOptions
    {
        public bool InstallShortcut;
        public string ShortcutDir;   // 仅与 --install-shortcut 同用，必须是绝对路径
        public bool NoShortcut;
        public string Error;         // 非 null 表示用法错误

        public bool IsValid { get { return Error == null; } }

        public static string Usage
        {
            get
            {
                return "Usage: RAG-Workbench.exe [--no-shortcut]\n" +
                       "       RAG-Workbench.exe --install-shortcut [--shortcut-dir <absolute dir>]\n" +
                       "Exit codes: 0 ok, 2 shortcut name conflict, 3 shortcut failed, 64 usage error.";
            }
        }

        public static LauncherOptions Parse(string[] args)
        {
            LauncherOptions o = new LauncherOptions();
            if (args == null) return o;
            for (int i = 0; i < args.Length; i++)
            {
                string a = args[i];
                if (a == "--install-shortcut")
                {
                    o.InstallShortcut = true;
                }
                else if (a == "--no-shortcut")
                {
                    o.NoShortcut = true;
                }
                else if (a == "--shortcut-dir")
                {
                    if (i + 1 >= args.Length || args[i + 1].StartsWith("--"))
                    {
                        o.Error = "--shortcut-dir requires a directory.";
                        return o;
                    }
                    if (o.ShortcutDir != null)
                    {
                        o.Error = "--shortcut-dir given more than once.";
                        return o;
                    }
                    o.ShortcutDir = args[++i];
                }
                else
                {
                    o.Error = "Unknown argument: " + a;
                    return o;
                }
            }
            if (o.ShortcutDir != null)
            {
                if (!o.InstallShortcut)
                    o.Error = "--shortcut-dir is only valid with --install-shortcut.";
                else if (!IsAbsoluteLocalPath(o.ShortcutDir))
                    o.Error = "--shortcut-dir must be an absolute path.";
            }
            if (o.Error == null && o.InstallShortcut && o.NoShortcut)
                o.Error = "--install-shortcut and --no-shortcut cannot be combined.";
            return o;
        }

        static bool IsAbsoluteLocalPath(string p)
        {
            if (string.IsNullOrEmpty(p)) return false;
            try
            {
                // "C:foo" 与 "\foo" 在 .NET 里 IsPathRooted 为 true，但不是完整绝对路径。
                if (!Path.IsPathRooted(p)) return false;
                if (p.Length >= 3 && char.IsLetter(p[0]) && p[1] == ':' && (p[2] == '\\' || p[2] == '/')) return true;
                if (p.StartsWith(@"\\")) return true;
                return false;
            }
            catch (ArgumentException)
            {
                return false;
            }
        }
    }

    // ------------------------------------------------------------------ 参数转义

    public static class WinArgs
    {
        // 按 CommandLineToArgvW / MSVCRT 规则转义单个参数，Python 的 sys.argv 使用同一规则。
        public static string Quote(string arg)
        {
            if (arg == null) arg = "";
            if (arg.Length > 0 && arg.IndexOfAny(new char[] { ' ', '\t', '\n', '\v', '"' }) < 0)
                return arg;

            StringBuilder sb = new StringBuilder(arg.Length + 2);
            sb.Append('"');
            int i = 0;
            while (i < arg.Length)
            {
                int backslashes = 0;
                while (i < arg.Length && arg[i] == '\\')
                {
                    backslashes++;
                    i++;
                }
                if (i == arg.Length)
                {
                    sb.Append('\\', backslashes * 2);   // 结尾引号前的反斜杠要成对
                    break;
                }
                if (arg[i] == '"')
                {
                    sb.Append('\\', backslashes * 2 + 1);
                    sb.Append('"');
                }
                else
                {
                    sb.Append('\\', backslashes);
                    sb.Append(arg[i]);
                }
                i++;
            }
            sb.Append('"');
            return sb.ToString();
        }

        public static string Join(IEnumerable<string> args)
        {
            StringBuilder sb = new StringBuilder();
            foreach (string a in args)
            {
                if (sb.Length > 0) sb.Append(' ');
                sb.Append(Quote(a));
            }
            return sb.ToString();
        }
    }

    // ------------------------------------------------------------------ 工作区身份

    public static class Workspace
    {
        // 与后端约定：绝对根路径去掉末尾分隔符，UTF-8 编码，SHA-256，取前 16 个小写十六进制字符。
        public static string NormalizeRoot(string root)
        {
            string full = Path.GetFullPath(root);
            string trimmed = full.TrimEnd('\\', '/');
            // "C:\" 去掉后变成 "C:"，仍按约定使用去掉分隔符后的形式。
            return trimmed.Length == 0 ? full : trimmed;
        }

        public static string ComputeId(string root)
        {
            byte[] bytes = Encoding.UTF8.GetBytes(NormalizeRoot(root).ToLowerInvariant());
            byte[] hash;
            using (SHA256 sha = SHA256.Create())
            {
                hash = sha.ComputeHash(bytes);
            }
            StringBuilder sb = new StringBuilder(16);
            for (int i = 0; i < 8; i++) sb.Append(hash[i].ToString("x2"));
            return sb.ToString();
        }
    }

    // ------------------------------------------------------------------ 服务健康

    public enum ServiceState
    {
        NotRunning,      // 端口无人监听
        Ours,            // 本项目（同一 workspace_id）的工作台
        OtherWorkspace,  // 另一个目录的工作台
        Occupied,        // 端口被其他程序占用，或响应无法识别
    }

    public sealed class HealthInfo
    {
        public ServiceState State;
        public bool Ready;
        public bool Busy;
        public string Message = "";
        public string WorkspaceId = "";

        public static HealthInfo NotRunning()
        {
            HealthInfo h = new HealthInfo();
            h.State = ServiceState.NotRunning;
            return h;
        }

        public static HealthInfo OccupiedBy(string message)
        {
            HealthInfo h = new HealthInfo();
            h.State = ServiceState.Occupied;
            h.Message = message ?? "";
            return h;
        }

        // 仅当 service 与 workspace_id 都匹配时才认定为“本项目的服务”。
        // 缺少 workspace_id 的旧版服务无法证明身份，按占用处理。
        public static HealthInfo Classify(string json, string expectedWorkspaceId)
        {
            Dictionary<string, object> d = JsonUtil.ParseObject(json);
            if (d == null) return OccupiedBy("端口上的程序返回了无法识别的响应。");

            HealthInfo h = new HealthInfo();
            h.Ready = JsonUtil.GetBool(d, "ready");
            h.Busy = JsonUtil.GetBool(d, "busy");
            h.Message = JsonUtil.GetString(d, "message");
            h.WorkspaceId = JsonUtil.GetString(d, "workspace_id");

            if (JsonUtil.GetString(d, "service") != AppInfo.HealthService)
            {
                h.State = ServiceState.Occupied;
                h.Message = "端口被其他程序占用。";
            }
            else if (h.WorkspaceId.Length == 0 ||
                     !string.Equals(h.WorkspaceId, expectedWorkspaceId, StringComparison.OrdinalIgnoreCase))
            {
                h.State = h.WorkspaceId.Length == 0 ? ServiceState.Occupied : ServiceState.OtherWorkspace;
            }
            else
            {
                h.State = ServiceState.Ours;
            }
            return h;
        }

        // 同步探测；在线程池里调用，不要在 UI 线程上调用。
        public static HealthInfo Probe(string expectedWorkspaceId, int timeoutMs)
        {
            try
            {
                HttpWebRequest req = (HttpWebRequest)WebRequest.Create(AppInfo.HealthUrl);
                req.Proxy = null;                       // 本机地址不走系统代理
                req.Timeout = timeoutMs;
                req.ReadWriteTimeout = timeoutMs;
                req.KeepAlive = false;
                req.Accept = "application/json";
                using (HttpWebResponse resp = (HttpWebResponse)req.GetResponse())
                using (StreamReader reader = new StreamReader(resp.GetResponseStream(), Encoding.UTF8))
                {
                    return Classify(reader.ReadToEnd(), expectedWorkspaceId);
                }
            }
            catch (WebException ex)
            {
                if (ex.Status == WebExceptionStatus.ConnectFailure) return NotRunning();
                if (ex.Response != null)
                {
                    ex.Response.Close();
                    return OccupiedBy("端口被其他程序占用。");
                }
                if (ex.Status == WebExceptionStatus.Timeout)
                    return OccupiedBy("端口有程序在监听，但没有及时响应。");
                // 连接已建立却读不到有效响应：有程序占着端口，不能当作空闲。
                return OccupiedBy("无法确认端口上的程序。");
            }
            catch (Exception)
            {
                return OccupiedBy("无法确认端口上的程序。");
            }
        }
    }

    // ------------------------------------------------------------------ JSON

    public static class JsonUtil
    {
        public static Dictionary<string, object> ParseObject(string json)
        {
            if (string.IsNullOrEmpty(json)) return null;
            try
            {
                JavaScriptSerializer s = new JavaScriptSerializer();
                return s.DeserializeObject(json.Trim()) as Dictionary<string, object>;
            }
            catch (Exception)
            {
                return null;
            }
        }

        public static bool GetBool(Dictionary<string, object> d, string key)
        {
            object v;
            return d.TryGetValue(key, out v) && v is bool && (bool)v;
        }

        public static bool HasBool(Dictionary<string, object> d, string key)
        {
            object v;
            return d.TryGetValue(key, out v) && v is bool;
        }

        public static string GetString(Dictionary<string, object> d, string key)
        {
            object v;
            if (d.TryGetValue(key, out v) && v is string) return (string)v;
            return "";
        }
    }

    // ------------------------------------------------------------------ 安装后端输出

    public sealed class CheckInfo
    {
        public bool PythonReady, NodeReady, ModelsReady, IndexReady, RequirementsReady, Ready;
        public string Message = "";

        // --check 约定只输出一个 JSON 对象。容忍前面混入的警告行：优先整体解析，失败再取最后一行 { 开头的内容。
        public static CheckInfo Parse(string stdout)
        {
            if (stdout == null) return null;
            Dictionary<string, object> d = JsonUtil.ParseObject(stdout);
            if (d == null)
            {
                string[] lines = stdout.Split(new char[] { '\r', '\n' }, StringSplitOptions.RemoveEmptyEntries);
                for (int i = lines.Length - 1; i >= 0 && d == null; i--)
                {
                    if (lines[i].TrimStart().StartsWith("{")) d = JsonUtil.ParseObject(lines[i]);
                }
            }
            if (d == null || !JsonUtil.HasBool(d, "ready")) return null;

            CheckInfo c = new CheckInfo();
            c.PythonReady = JsonUtil.GetBool(d, "python_ready");
            c.NodeReady = JsonUtil.GetBool(d, "node_ready");
            c.ModelsReady = JsonUtil.GetBool(d, "models_ready");
            c.IndexReady = JsonUtil.GetBool(d, "index_ready");
            c.RequirementsReady = JsonUtil.GetBool(d, "requirements_ready");
            c.Ready = JsonUtil.GetBool(d, "ready");
            c.Message = JsonUtil.GetString(d, "message");
            return c;
        }

        public bool EnvironmentReady { get { return PythonReady && NodeReady && RequirementsReady; } }
        public bool DataReady { get { return ModelsReady && IndexReady; } }
    }

    public sealed class StageLine
    {
        public string Key;
        public string Message;
    }

    public static class SetupOutput
    {
        // 格式：STAGE|key|message；message 内可以再含 '|'。
        public static StageLine ParseStage(string line)
        {
            if (line == null || !line.StartsWith("STAGE|")) return null;
            int second = line.IndexOf('|', 6);
            if (second < 0) return null;
            string key = line.Substring(6, second - 6).Trim();
            if (key.Length == 0) return null;
            StageLine s = new StageLine();
            s.Key = key;
            s.Message = line.Substring(second + 1).Trim();
            return s;
        }

        static readonly Regex SecretLike = new Regex(
            @"(sk-[A-Za-z0-9_\-]{12,})|((api[_-]?key|token|secret|password)\s*[:=]\s*\S+)|(authorization\s*[:=].*)",
            RegexOptions.IgnoreCase | RegexOptions.CultureInvariant);

        // 安装日志不应含密钥；这里再兜底遮蔽形似密钥的片段，防止第三方工具回显。
        public static string Redact(string line)
        {
            if (string.IsNullOrEmpty(line)) return line;
            return SecretLike.Replace(line, "[已遮蔽]");
        }

        public static string Shorten(string text, int max)
        {
            if (text == null) return "";
            text = text.Trim();
            return text.Length <= max ? text : text.Substring(0, max - 1) + "…";
        }
    }

    // ------------------------------------------------------------------ Python 解析

    public sealed class PythonCommand
    {
        public string FileName;
        public List<string> PrefixArgs = new List<string>();
        public bool UsesPyLauncher;

        // 项目 .venv 优先；否则用 py 启动器指定 3.12。不会搜索或下载其他 Python。
        public static PythonCommand Resolve(string root)
        {
            PythonCommand p = new PythonCommand();
            string venv = Path.Combine(root, @".venv\Scripts\python.exe");
            if (File.Exists(venv))
            {
                p.FileName = venv;
            }
            else
            {
                p.FileName = "py";
                p.PrefixArgs.Add("-3.12");
                p.UsesPyLauncher = true;
            }
            p.PrefixArgs.Add("-X");
            p.PrefixArgs.Add("utf8");
            return p;
        }

        public List<string> With(params string[] args)
        {
            List<string> all = new List<string>(PrefixArgs);
            all.AddRange(args);
            return all;
        }
    }

    // ------------------------------------------------------------------ 桌面快捷方式

    public enum ShortcutStatus { Created, Repaired, Unchanged, Conflict, Failed }

    public sealed class ShortcutResult
    {
        public ShortcutStatus Status;
        public string Path = "";
        public string Message = "";

        public bool Succeeded
        {
            get { return Status == ShortcutStatus.Created || Status == ShortcutStatus.Repaired || Status == ShortcutStatus.Unchanged; }
        }

        public int ExitCode
        {
            get
            {
                if (Succeeded) return ExitCodes.Ok;
                return Status == ShortcutStatus.Conflict ? ExitCodes.ShortcutConflict : ExitCodes.ShortcutFailed;
            }
        }

        internal static ShortcutResult Make(ShortcutStatus s, string path, string message)
        {
            ShortcutResult r = new ShortcutResult();
            r.Status = s;
            r.Path = path ?? "";
            r.Message = message ?? "";
            return r;
        }
    }

    public sealed class ShortcutInfo
    {
        public string TargetPath = "";
        public string Arguments = "";
        public string WorkingDirectory = "";
        public string IconLocation = "";
    }

    public static class ShortcutManager
    {
        public static string DesktopDirectory()
        {
            // DesktopDirectory 走 SHGetFolderPath，已包含 OneDrive 与文件夹重定向。
            return Environment.GetFolderPath(Environment.SpecialFolder.DesktopDirectory);
        }

        public static string PrimaryName() { return AppInfo.ShortcutBaseName + ".lnk"; }

        public static string AlternateName(string root)
        {
            string folder = System.IO.Path.GetFileName(Workspace.NormalizeRoot(root));
            if (string.IsNullOrEmpty(folder)) folder = Workspace.ComputeId(root).Substring(0, 6);
            foreach (char c in System.IO.Path.GetInvalidFileNameChars()) folder = folder.Replace(c, '_');
            return AppInfo.ShortcutBaseName + " (" + folder + ").lnk";
        }

        public static string TargetFor(string root) { return System.IO.Path.Combine(Workspace.NormalizeRoot(root), AppInfo.ExeName); }
        public static string IconFor(string root) { return System.IO.Path.Combine(Workspace.NormalizeRoot(root), @"assets\app.ico"); }

        enum Slot { Free, Ours, OursStale, Foreign }

        // 候选名称：“RAG 工作台.lnk”，其次“RAG 工作台 (目录名).lnk”。先看两个名称的现状再决定，避免重复：
        //   1. 任一名称已指向本项目 → 设置一致则不动，不一致则原地修复（优先主名称）；
        //   2. 任一名称指向一个已不存在的 RAG-Workbench.exe（项目被移动）→ 原地修复；
        //   3. 否则写到第一个空闲名称；
        //   4. 两个名称都被其他有效目标占用 → 冲突，不写任何文件。
        public static ShortcutResult Ensure(string root, string shortcutDir)
        {
            string target = TargetFor(root);
            try
            {
                if (string.IsNullOrEmpty(shortcutDir) || !Directory.Exists(shortcutDir))
                    return ShortcutResult.Make(ShortcutStatus.Failed, shortcutDir, "快捷方式目录不存在。");

                string[] paths = new string[]
                {
                    System.IO.Path.Combine(shortcutDir, PrimaryName()),
                    System.IO.Path.Combine(shortcutDir, AlternateName(root)),
                };
                Slot[] slots = new Slot[paths.Length];
                ShortcutInfo[] infos = new ShortcutInfo[paths.Length];
                for (int i = 0; i < paths.Length; i++) slots[i] = Inspect(paths[i], target, out infos[i]);

                for (int i = 0; i < paths.Length; i++)
                    if (slots[i] == Slot.Ours && Matches(infos[i], root))
                        return ShortcutResult.Make(ShortcutStatus.Unchanged, paths[i], "桌面图标正常。");

                foreach (Slot wanted in new Slot[] { Slot.Ours, Slot.OursStale })
                {
                    for (int i = 0; i < paths.Length; i++)
                    {
                        if (slots[i] != wanted) continue;
                        Write(paths[i], root);
                        return ShortcutResult.Make(ShortcutStatus.Repaired, paths[i], "已修复桌面图标。");
                    }
                }

                for (int i = 0; i < paths.Length; i++)
                {
                    if (slots[i] != Slot.Free) continue;
                    Write(paths[i], root);
                    return ShortcutResult.Make(ShortcutStatus.Created, paths[i], "已创建桌面图标。");
                }

                return ShortcutResult.Make(ShortcutStatus.Conflict, paths[0],
                    "桌面上已有同名快捷方式指向其他程序，未覆盖。");
            }
            catch (Exception ex)
            {
                return ShortcutResult.Make(ShortcutStatus.Failed, shortcutDir, "无法创建桌面图标：" + SetupOutput.Shorten(ex.Message, 120));
            }
        }

        static Slot Inspect(string path, string ourTarget, out ShortcutInfo info)
        {
            info = null;
            if (!File.Exists(path)) return Directory.Exists(path) ? Slot.Foreign : Slot.Free;
            try
            {
                info = Read(path);
            }
            catch (Exception)
            {
                return Slot.Foreign;   // 读不出来的 .lnk 不是我们能安全改写的
            }
            if (SamePath(info.TargetPath, ourTarget)) return Slot.Ours;
            if (info.TargetPath.Length > 0 && !File.Exists(info.TargetPath) && !Directory.Exists(info.TargetPath) &&
                string.Equals(System.IO.Path.GetFileName(info.TargetPath), AppInfo.ExeName, StringComparison.OrdinalIgnoreCase))
                return Slot.OursStale;
            return Slot.Foreign;
        }

        static bool Matches(ShortcutInfo s, string root)
        {
            string icon = s.IconLocation;
            int comma = icon.LastIndexOf(',');
            if (comma > 0) icon = icon.Substring(0, comma);
            return SamePath(s.TargetPath, TargetFor(root)) &&
                   SamePath(s.WorkingDirectory, Workspace.NormalizeRoot(root)) &&
                   SamePath(icon, IconFor(root)) &&
                   s.Arguments.Length == 0;
        }

        public static bool SamePath(string a, string b)
        {
            if (string.IsNullOrEmpty(a) || string.IsNullOrEmpty(b)) return false;
            try
            {
                return string.Equals(System.IO.Path.GetFullPath(a).TrimEnd('\\'), System.IO.Path.GetFullPath(b).TrimEnd('\\'),
                    StringComparison.OrdinalIgnoreCase);
            }
            catch (Exception)
            {
                return false;
            }
        }

        public static ShortcutInfo Read(string path)
        {
            object shell = null, link = null;
            try
            {
                shell = CreateShell();
                link = Invoke(shell, "CreateShortcut", path);
                ShortcutInfo info = new ShortcutInfo();
                info.TargetPath = (Get(link, "TargetPath") as string) ?? "";
                info.Arguments = (Get(link, "Arguments") as string) ?? "";
                info.WorkingDirectory = (Get(link, "WorkingDirectory") as string) ?? "";
                info.IconLocation = (Get(link, "IconLocation") as string) ?? "";
                return info;
            }
            finally
            {
                Release(link);
                Release(shell);
            }
        }

        static void Write(string path, string root)
        {
            WriteLink(path, TargetFor(root), Workspace.NormalizeRoot(root), IconFor(root) + ",0");
        }

        // 原样写一个 .lnk（测试用它布置“别人的”快捷方式）。
        public static void WriteLink(string path, string target, string workingDirectory, string iconLocation)
        {
            object shell = null, link = null;
            try
            {
                shell = CreateShell();
                link = Invoke(shell, "CreateShortcut", path);
                Set(link, "TargetPath", target);
                Set(link, "Arguments", "");
                Set(link, "WorkingDirectory", workingDirectory);
                Set(link, "IconLocation", iconLocation);
                Set(link, "Description", "资料问答工作台");
                Invoke(link, "Save");
            }
            finally
            {
                Release(link);
                Release(shell);
            }
        }

        static object CreateShell()
        {
            Type t = Type.GetTypeFromProgID("WScript.Shell", false);
            if (t == null) throw new InvalidOperationException("系统未提供 WScript.Shell。");
            return Activator.CreateInstance(t);
        }

        static object Invoke(object target, string method, params object[] args)
        {
            try
            {
                return target.GetType().InvokeMember(method, BindingFlags.InvokeMethod, null, target, args);
            }
            catch (TargetInvocationException ex)
            {
                throw ex.InnerException ?? ex;
            }
        }

        static object Get(object target, string prop)
        {
            return target.GetType().InvokeMember(prop, BindingFlags.GetProperty, null, target, null);
        }

        static void Set(object target, string prop, object value)
        {
            target.GetType().InvokeMember(prop, BindingFlags.SetProperty, null, target, new object[] { value });
        }

        static void Release(object com)
        {
            if (com != null && Marshal.IsComObject(com)) Marshal.FinalReleaseComObject(com);
        }
    }

    // 记录“已为哪个根目录自动处理过桌面图标”。只在首次（或项目被移动后）自动创建；
    // 用户删掉图标后不再每次偷偷放回，想恢复可点“修复桌面图标”。
    public static class ShortcutState
    {
        public static string StatePath(string root) { return Path.Combine(root, @"data\ui\launcher-state.json"); }

        public static bool ShouldAutoEnsure(string statePath, string root)
        {
            try
            {
                if (!File.Exists(statePath)) return true;
                Dictionary<string, object> d = JsonUtil.ParseObject(File.ReadAllText(statePath, Encoding.UTF8));
                if (d == null) return true;
                return JsonUtil.GetString(d, "shortcut_workspace_id") != Workspace.ComputeId(root);
            }
            catch (Exception)
            {
                return true;
            }
        }

        public static void MarkDone(string statePath, string root, ShortcutResult result)
        {
            Dictionary<string, object> d = new Dictionary<string, object>();
            d["shortcut_workspace_id"] = Workspace.ComputeId(root);
            d["shortcut_path"] = result.Path;
            d["shortcut_status"] = result.Status.ToString();
            d["updated"] = DateTime.Now.ToString("s");
            Directory.CreateDirectory(Path.GetDirectoryName(statePath));
            File.WriteAllText(statePath, new JavaScriptSerializer().Serialize(d), new UTF8Encoding(false));
        }
    }

    // ------------------------------------------------------------------ 进程

    // Job Object：安装进程及其子孙都在其中。终止时只结束这个 Job，不按进程名查杀。
    public sealed class OwnedJob : IDisposable
    {
        [StructLayout(LayoutKind.Sequential)]
        struct JOBOBJECT_BASIC_LIMIT_INFORMATION
        {
            public long PerProcessUserTimeLimit;
            public long PerJobUserTimeLimit;
            public uint LimitFlags;
            public UIntPtr MinimumWorkingSetSize;
            public UIntPtr MaximumWorkingSetSize;
            public uint ActiveProcessLimit;
            public UIntPtr Affinity;
            public uint PriorityClass;
            public uint SchedulingClass;
        }

        [StructLayout(LayoutKind.Sequential)]
        struct IO_COUNTERS
        {
            public ulong ReadOperationCount, WriteOperationCount, OtherOperationCount;
            public ulong ReadTransferCount, WriteTransferCount, OtherTransferCount;
        }

        [StructLayout(LayoutKind.Sequential)]
        struct JOBOBJECT_EXTENDED_LIMIT_INFORMATION
        {
            public JOBOBJECT_BASIC_LIMIT_INFORMATION BasicLimitInformation;
            public IO_COUNTERS IoInfo;
            public UIntPtr ProcessMemoryLimit;
            public UIntPtr JobMemoryLimit;
            public UIntPtr PeakProcessMemoryUsed;
            public UIntPtr PeakJobMemoryUsed;
        }

        const int JobObjectExtendedLimitInformation = 9;
        const uint JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x2000;

        [DllImport("kernel32.dll", CharSet = CharSet.Unicode, SetLastError = true)]
        static extern IntPtr CreateJobObject(IntPtr attributes, string name);
        [DllImport("kernel32.dll", SetLastError = true)]
        static extern bool SetInformationJobObject(IntPtr job, int infoClass, ref JOBOBJECT_EXTENDED_LIMIT_INFORMATION info, uint length);
        [DllImport("kernel32.dll", SetLastError = true)]
        static extern bool AssignProcessToJobObject(IntPtr job, IntPtr process);
        [DllImport("kernel32.dll", SetLastError = true)]
        static extern bool TerminateJobObject(IntPtr job, uint exitCode);
        [DllImport("kernel32.dll", SetLastError = true)]
        static extern bool CloseHandle(IntPtr handle);

        IntPtr handle;

        public OwnedJob()
        {
            handle = CreateJobObject(IntPtr.Zero, null);
            if (handle == IntPtr.Zero) throw new Win32Exception(Marshal.GetLastWin32Error());
            JOBOBJECT_EXTENDED_LIMIT_INFORMATION info = new JOBOBJECT_EXTENDED_LIMIT_INFORMATION();
            info.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE;
            if (!SetInformationJobObject(handle, JobObjectExtendedLimitInformation, ref info,
                    (uint)Marshal.SizeOf(typeof(JOBOBJECT_EXTENDED_LIMIT_INFORMATION))))
            {
                int err = Marshal.GetLastWin32Error();
                CloseHandle(handle);
                handle = IntPtr.Zero;
                throw new Win32Exception(err);
            }
        }

        public bool Assign(Process p)
        {
            return handle != IntPtr.Zero && AssignProcessToJobObject(handle, p.Handle);
        }

        public void Terminate()
        {
            if (handle != IntPtr.Zero) TerminateJobObject(handle, 1);
        }

        public void Dispose()
        {
            if (handle != IntPtr.Zero)
            {
                CloseHandle(handle);
                handle = IntPtr.Zero;
            }
        }
    }

    public sealed class ProcessOutcome
    {
        public int ExitCode = -1;
        public bool StartFailed;
        public bool TimedOut;
        public bool Cancelled;
        public string StartError = "";
        public string Stdout = "";
        public string LastErrorLine = "";
    }

    // 隐藏窗口、异步读 stdout/stderr、不经过 cmd。回调在线程池线程上触发，界面层自行 BeginInvoke。
    public sealed class ProcessRunner
    {
        readonly string fileName;
        readonly List<string> args;
        readonly string workingDir;
        readonly bool ownTree;
        readonly int timeoutMs;
        Process process;
        OwnedJob job;
        int cancelled;

        public event Action<string, bool> Line;          // (文本, 是否 stderr)
        public event Action<ProcessOutcome> Completed;

        public ProcessRunner(string fileName, List<string> args, string workingDir, bool ownTree, int timeoutMs)
        {
            this.fileName = fileName;
            this.args = args;
            this.workingDir = workingDir;
            this.ownTree = ownTree;
            this.timeoutMs = timeoutMs;
        }

        public static ProcessStartInfo BuildStartInfo(string fileName, IEnumerable<string> args, string workingDir)
        {
            ProcessStartInfo psi = new ProcessStartInfo(fileName, WinArgs.Join(args));
            psi.WorkingDirectory = workingDir;
            psi.UseShellExecute = false;
            psi.CreateNoWindow = true;
            psi.WindowStyle = ProcessWindowStyle.Hidden;
            psi.RedirectStandardOutput = true;
            psi.RedirectStandardError = true;
            psi.StandardOutputEncoding = new UTF8Encoding(false);
            psi.StandardErrorEncoding = new UTF8Encoding(false);
            psi.EnvironmentVariables["PYTHONUTF8"] = "1";
            psi.EnvironmentVariables["PYTHONIOENCODING"] = "utf-8";
            return psi;
        }

        public void Start()
        {
            ThreadPool.QueueUserWorkItem(delegate { Run(); });
        }

        public void Cancel()
        {
            Interlocked.Exchange(ref cancelled, 1);
            try
            {
                if (job != null) job.Terminate();
                else if (process != null && !process.HasExited) process.Kill();   // 只结束自己启动的这一个进程
            }
            catch (Exception) { }
        }

        void Run()
        {
            ProcessOutcome outcome = new ProcessOutcome();
            StringBuilder stdout = new StringBuilder();
            ManualResetEvent outDone = new ManualResetEvent(false);
            ManualResetEvent errDone = new ManualResetEvent(false);
            object gate = new object();
            bool started = false;
            try
            {
                process = new Process();
                process.StartInfo = BuildStartInfo(fileName, args, workingDir);
                process.OutputDataReceived += delegate(object s, DataReceivedEventArgs e)
                {
                    if (e.Data == null) { outDone.Set(); return; }
                    lock (gate) stdout.AppendLine(e.Data);
                    Raise(e.Data, false);
                };
                process.ErrorDataReceived += delegate(object s, DataReceivedEventArgs e)
                {
                    if (e.Data == null) { errDone.Set(); return; }
                    if (e.Data.Trim().Length > 0) outcome.LastErrorLine = e.Data;
                    Raise(e.Data, true);
                };
                if (ownTree) job = new OwnedJob();
                try
                {
                    process.Start();
                    started = true;
                }
                catch (Win32Exception ex)
                {
                    outcome.StartFailed = true;
                    outcome.StartError = ex.Message;
                    return;
                }
                if (job != null && !job.Assign(process))
                {
                    // 无法放进 Job 时退回为只结束直接子进程，不扩大终止范围。
                    job.Dispose();
                    job = null;
                }
                if (cancelled != 0) Cancel();
                process.BeginOutputReadLine();
                process.BeginErrorReadLine();

                if (!process.WaitForExit(timeoutMs <= 0 ? int.MaxValue : timeoutMs))
                {
                    outcome.TimedOut = true;
                    Cancel();
                    process.WaitForExit(5000);
                }
                // 孙进程可能继承了管道句柄；最多再等 3 秒读完剩余输出，不无限阻塞。
                WaitHandle.WaitAll(new WaitHandle[] { outDone, errDone }, 3000);
                outcome.ExitCode = process.HasExited ? process.ExitCode : -1;
            }
            catch (Exception ex)
            {
                outcome.StartFailed = !started;
                outcome.StartError = ex.Message;
            }
            finally
            {
                lock (gate) outcome.Stdout = stdout.ToString();
                outcome.Cancelled = cancelled != 0 && !outcome.TimedOut;
                if (job != null) { job.Dispose(); job = null; }
                if (process != null) process.Dispose();
                Action<ProcessOutcome> done = Completed;
                if (done != null) done(outcome);
            }
        }

        void Raise(string text, bool isError)
        {
            Action<string, bool> h = Line;
            if (h != null) h(text, isError);
        }
    }

    // ------------------------------------------------------------------ 界面

    static class Palette
    {
        public static readonly Color Page = Color.FromArgb(0xF7, 0xF8, 0xFA);
        public static readonly Color Card = Color.White;
        public static readonly Color Ink = Color.FromArgb(0x1D, 0x23, 0x2A);
        public static readonly Color Body = Color.FromArgb(0x49, 0x51, 0x5C);
        public static readonly Color Sub = Color.FromArgb(0x5E, 0x68, 0x75);
        public static readonly Color Muted = Color.FromArgb(0x73, 0x80, 0x8B);
        public static readonly Color Line = Color.FromArgb(0xE1, 0xE5, 0xE9);
        public static readonly Color FieldLine = Color.FromArgb(0xD6, 0xDC, 0xE2);
        public static readonly Color Selected = Color.FromArgb(0xE8, 0xEC, 0xF1);
        public static readonly Color Dot = Color.FromArgb(0x97, 0xA2, 0xAD);
        public static readonly Color Ok = Color.FromArgb(0x2F, 0x7D, 0x4F);
        public static readonly Color Warn = Color.FromArgb(0x9A, 0x6A, 0x12);
        public static readonly Color Err = Color.FromArgb(0xB3, 0x39, 0x2F);
    }

    enum Tone { Neutral, Ok, Warn, Err }

    sealed class CardPanel : Panel
    {
        readonly Func<int, int> s;
        public CardPanel(Func<int, int> scale)
        {
            s = scale;
            DoubleBuffered = true;
            ResizeRedraw = true;
            BackColor = Palette.Page;
        }

        protected override void OnPaint(PaintEventArgs e)
        {
            base.OnPaint(e);
            e.Graphics.SmoothingMode = SmoothingMode.AntiAlias;
            Rectangle r = new Rectangle(0, 0, Width - 1, Height - 1);
            using (GraphicsPath path = Rounded(r, s(10)))
            using (SolidBrush fill = new SolidBrush(Palette.Card))
            using (Pen pen = new Pen(Palette.Line, 1))
            {
                e.Graphics.FillPath(fill, path);
                e.Graphics.DrawPath(pen, path);
            }
        }

        internal static GraphicsPath Rounded(Rectangle r, int radius)
        {
            int d = Math.Max(2, radius * 2);
            GraphicsPath p = new GraphicsPath();
            p.AddArc(r.X, r.Y, d, d, 180, 90);
            p.AddArc(r.Right - d, r.Y, d, d, 270, 90);
            p.AddArc(r.Right - d, r.Bottom - d, d, d, 0, 90);
            p.AddArc(r.X, r.Bottom - d, d, d, 90, 90);
            p.CloseFigure();
            return p;
        }
    }

    sealed class StatusDot : Control
    {
        Color color = Palette.Dot;
        public StatusDot() { SetStyle(ControlStyles.SupportsTransparentBackColor | ControlStyles.OptimizedDoubleBuffer | ControlStyles.AllPaintingInWmPaint | ControlStyles.UserPaint, true); }
        public Color DotColor { get { return color; } set { color = value; Invalidate(); } }
        protected override void OnPaint(PaintEventArgs e)
        {
            e.Graphics.SmoothingMode = SmoothingMode.AntiAlias;
            int d = Math.Min(Width, Height) - 2;
            using (SolidBrush b = new SolidBrush(color))
                e.Graphics.FillEllipse(b, (Width - d) / 2f, (Height - d) / 2f, d, d);
        }
    }

    // 细进度条：忙碌时一段墨色滑块往返，空闲时只显示发丝线。
    sealed class BusyStrip : Control
    {
        readonly System.Windows.Forms.Timer timer = new System.Windows.Forms.Timer();
        float phase;
        bool active;

        public BusyStrip()
        {
            SetStyle(ControlStyles.OptimizedDoubleBuffer | ControlStyles.AllPaintingInWmPaint | ControlStyles.UserPaint | ControlStyles.ResizeRedraw, true);
            timer.Interval = 30;
            timer.Tick += delegate { phase = (phase + 0.012f) % 1f; Invalidate(); };
        }

        public bool Active
        {
            get { return active; }
            set { active = value; timer.Enabled = value; Invalidate(); }
        }

        protected override void OnPaint(PaintEventArgs e)
        {
            e.Graphics.Clear(BackColor);
            using (SolidBrush track = new SolidBrush(Palette.Line))
                e.Graphics.FillRectangle(track, 0, 0, Width, Height);
            if (!active) return;
            float seg = Width * 0.28f;
            float x = (Width + seg) * phase - seg;
            using (SolidBrush b = new SolidBrush(Palette.Ink))
                e.Graphics.FillRectangle(b, x, 0, seg, Height);
        }

        protected override void Dispose(bool disposing)
        {
            if (disposing) timer.Dispose();
            base.Dispose(disposing);
        }
    }

    sealed class LauncherForm : Form
    {
        readonly string root;
        readonly string workspaceId;
        readonly bool noShortcut;
        readonly float dpiScale;

        // 状态
        CheckInfo lastCheck;
        string checkProblem;          // 检查无法运行时的说明（缺 Python、缺后端脚本等）
        bool checkPythonMissing;
        HealthInfo lastHealth = HealthInfo.NotRunning();
        string operation;             // null 表示空闲；否则为当前操作名
        ProcessRunner setupRunner;
        bool closeConfirmed;
        int probing;
        string idleHint;              // 空闲时在忙碌条下方显示的一行轻提示（如自动创建桌面图标的结果）

        // 控件
        Label envValue, envDetail, dataValue, dataDetail, svcValue, svcDetail;
        StatusDot envDot, dataDot, svcDot;
        BusyStrip strip;
        Label stripLabel;
        Button openButton, setupButton, stopButton, refreshButton, helpButton, shortcutButton;
        Label noticeLabel;
        LinkLabel pythonLink, nodeLink;
        readonly System.Windows.Forms.Timer healthTimer = new System.Windows.Forms.Timer();

        public LauncherForm(string root, bool noShortcut)
        {
            this.root = root;
            this.noShortcut = noShortcut;
            workspaceId = Workspace.ComputeId(root);
            using (Graphics g = CreateGraphics()) dpiScale = g.DpiX / 96f;

            Text = "资料问答工作台";
            AutoScaleMode = AutoScaleMode.None;   // 尺寸全部由 S() 按 DPI 计算，字体用磅值自然缩放
            BackColor = Palette.Page;
            ForeColor = Palette.Ink;
            Font = new Font("Microsoft YaHei UI", 9.5f, FontStyle.Regular, GraphicsUnit.Point);
            StartPosition = FormStartPosition.CenterScreen;
            DoubleBuffered = true;

            Rectangle work = Screen.PrimaryScreen.WorkingArea;
            Size min = new Size(Math.Min(S(820), work.Width), Math.Min(S(580), work.Height));
            MinimumSize = min;
            Size = new Size(Math.Min(S(880), work.Width), Math.Min(S(620), work.Height));

            string ico = Path.Combine(root, @"assets\app.ico");
            try
            {
                if (File.Exists(ico)) Icon = new Icon(ico);
                else Icon = Icon.ExtractAssociatedIcon(Application.ExecutablePath);
            }
            catch (Exception) { }

            BuildUi();

            healthTimer.Interval = 5000;
            healthTimer.Tick += delegate { if (operation == null) ProbeHealth(null); };
        }

        int S(int px) { return (int)Math.Round(px * dpiScale); }

        Font F(float pt, FontStyle style) { return new Font("Microsoft YaHei UI", pt, style, GraphicsUnit.Point); }

        // ---------------- 布局

        void BuildUi()
        {
            TableLayoutPanel page = new TableLayoutPanel();
            page.Dock = DockStyle.Fill;
            page.BackColor = Palette.Page;
            page.Padding = new Padding(S(32), S(26), S(32), S(18));
            page.ColumnCount = 1;
            page.RowCount = 6;
            page.ColumnStyles.Add(new ColumnStyle(SizeType.Percent, 100));
            page.RowStyles.Add(new RowStyle(SizeType.Absolute, S(76)));   // 标题
            page.RowStyles.Add(new RowStyle(SizeType.Absolute, S(158)));  // 三张状态卡
            page.RowStyles.Add(new RowStyle(SizeType.Absolute, S(46)));   // 忙碌条
            page.RowStyles.Add(new RowStyle(SizeType.Absolute, S(60)));   // 主操作
            page.RowStyles.Add(new RowStyle(SizeType.Percent, 100));      // 提示
            page.RowStyles.Add(new RowStyle(SizeType.Absolute, S(74)));   // 页脚
            Controls.Add(page);

            page.Controls.Add(BuildHeader(), 0, 0);
            page.Controls.Add(BuildCards(), 0, 1);
            page.Controls.Add(BuildStrip(), 0, 2);
            page.Controls.Add(BuildActions(), 0, 3);
            page.Controls.Add(BuildNotice(), 0, 4);
            page.Controls.Add(BuildFooter(), 0, 5);
        }

        Control BuildHeader()
        {
            TableLayoutPanel t = new TableLayoutPanel();
            t.Dock = DockStyle.Fill;
            t.Margin = new Padding(0);
            t.ColumnCount = 2;
            t.RowCount = 2;
            t.ColumnStyles.Add(new ColumnStyle(SizeType.Absolute, S(64)));
            t.ColumnStyles.Add(new ColumnStyle(SizeType.Percent, 100));
            t.RowStyles.Add(new RowStyle(SizeType.Percent, 55));
            t.RowStyles.Add(new RowStyle(SizeType.Percent, 45));

            PictureBox pic = new PictureBox();
            pic.Size = new Size(S(48), S(48));
            pic.SizeMode = PictureBoxSizeMode.Zoom;
            pic.Anchor = AnchorStyles.Left;
            pic.Margin = new Padding(0, 0, S(12), 0);
            pic.Image = LoadLogo(S(48));
            t.Controls.Add(pic, 0, 0);
            t.SetRowSpan(pic, 2);

            Label title = new Label();
            title.Text = "资料问答工作台";
            title.Font = F(17f, FontStyle.Bold);
            title.ForeColor = Palette.Ink;
            title.AutoSize = true;
            title.Anchor = AnchorStyles.Left | AnchorStyles.Bottom;
            title.Margin = new Padding(0);
            t.Controls.Add(title, 1, 0);

            Label sub = new Label();
            sub.Text = "在本机检索你的资料，回答问题并标出引用来源。这里负责准备环境和启动服务。";
            sub.Font = F(9.5f, FontStyle.Regular);
            sub.ForeColor = Palette.Sub;
            sub.AutoEllipsis = true;
            sub.Dock = DockStyle.Fill;
            sub.Margin = new Padding(S(2), S(4), 0, 0);
            t.Controls.Add(sub, 1, 1);
            return t;
        }

        Image LoadLogo(int size)
        {
            try
            {
                string png = Path.Combine(root, @"assets\app.png");
                if (File.Exists(png))
                {
                    // 读入内存再解码，避免锁住文件。
                    using (MemoryStream ms = new MemoryStream(File.ReadAllBytes(png)))
                    using (Image src = Image.FromStream(ms))
                    {
                        Bitmap bmp = new Bitmap(size, size);
                        using (Graphics g = Graphics.FromImage(bmp))
                        {
                            g.InterpolationMode = InterpolationMode.HighQualityBicubic;
                            g.PixelOffsetMode = PixelOffsetMode.HighQuality;
                            g.DrawImage(src, 0, 0, size, size);
                        }
                        return bmp;
                    }
                }
                if (Icon != null) return new Icon(Icon, size, size).ToBitmap();
            }
            catch (Exception) { }
            return null;
        }

        Control BuildCards()
        {
            TableLayoutPanel t = new TableLayoutPanel();
            t.Dock = DockStyle.Fill;
            t.Margin = new Padding(0, S(6), 0, S(6));
            t.ColumnCount = 3;
            t.RowCount = 1;
            for (int i = 0; i < 3; i++) t.ColumnStyles.Add(new ColumnStyle(SizeType.Percent, 33.333f));
            t.RowStyles.Add(new RowStyle(SizeType.Percent, 100));

            t.Controls.Add(BuildCard("运行环境", out envDot, out envValue, out envDetail, new Padding(0, 0, S(8), 0)), 0, 0);
            t.Controls.Add(BuildCard("资料与模型", out dataDot, out dataValue, out dataDetail, new Padding(S(4), 0, S(4), 0)), 1, 0);
            t.Controls.Add(BuildCard("本机服务", out svcDot, out svcValue, out svcDetail, new Padding(S(8), 0, 0, 0)), 2, 0);
            return t;
        }

        Control BuildCard(string heading, out StatusDot dot, out Label value, out Label detail, Padding margin)
        {
            CardPanel card = new CardPanel(S);
            card.Dock = DockStyle.Fill;
            card.Margin = margin;
            card.Padding = new Padding(S(18), S(16), S(16), S(12));

            TableLayoutPanel t = new TableLayoutPanel();
            t.Dock = DockStyle.Fill;
            t.BackColor = Palette.Card;
            t.ColumnCount = 2;
            t.RowCount = 3;
            t.ColumnStyles.Add(new ColumnStyle(SizeType.Absolute, S(18)));
            t.ColumnStyles.Add(new ColumnStyle(SizeType.Percent, 100));
            t.RowStyles.Add(new RowStyle(SizeType.Absolute, S(26)));
            t.RowStyles.Add(new RowStyle(SizeType.Absolute, S(34)));
            t.RowStyles.Add(new RowStyle(SizeType.Percent, 100));

            Label h = new Label();
            h.Text = heading;
            h.Font = F(9f, FontStyle.Regular);
            h.ForeColor = Palette.Muted;
            h.Dock = DockStyle.Fill;
            h.Margin = new Padding(0);
            t.Controls.Add(h, 0, 0);
            t.SetColumnSpan(h, 2);

            dot = new StatusDot();
            dot.Size = new Size(S(10), S(10));
            dot.Anchor = AnchorStyles.Left;
            dot.Margin = new Padding(0, S(2), 0, 0);
            dot.BackColor = Palette.Card;
            t.Controls.Add(dot, 0, 1);

            value = new Label();
            value.Text = "检查中…";
            value.Font = F(12.5f, FontStyle.Bold);
            value.ForeColor = Palette.Ink;
            value.Dock = DockStyle.Fill;
            value.TextAlign = ContentAlignment.MiddleLeft;
            value.AutoEllipsis = true;
            value.Margin = new Padding(0);
            t.Controls.Add(value, 1, 1);

            detail = new Label();
            detail.Text = "";
            detail.Font = F(9f, FontStyle.Regular);
            detail.ForeColor = Palette.Sub;
            detail.Dock = DockStyle.Fill;
            detail.AutoEllipsis = true;
            detail.Margin = new Padding(0, S(6), 0, 0);
            t.Controls.Add(detail, 0, 2);
            t.SetColumnSpan(detail, 2);

            card.Controls.Add(t);
            return card;
        }

        Control BuildStrip()
        {
            TableLayoutPanel t = new TableLayoutPanel();
            t.Dock = DockStyle.Fill;
            t.Margin = new Padding(0, S(6), 0, 0);
            t.ColumnCount = 1;
            t.RowCount = 2;
            t.RowStyles.Add(new RowStyle(SizeType.Absolute, S(3)));
            t.RowStyles.Add(new RowStyle(SizeType.Percent, 100));

            strip = new BusyStrip();
            strip.Dock = DockStyle.Fill;
            strip.Margin = new Padding(0);
            strip.BackColor = Palette.Page;
            t.Controls.Add(strip, 0, 0);

            stripLabel = new Label();
            stripLabel.Dock = DockStyle.Fill;
            stripLabel.Margin = new Padding(0, S(6), 0, 0);
            stripLabel.Font = F(9f, FontStyle.Regular);
            stripLabel.ForeColor = Palette.Muted;
            stripLabel.AutoEllipsis = true;
            stripLabel.Text = "";
            t.Controls.Add(stripLabel, 0, 1);
            return t;
        }

        Control BuildActions()
        {
            TableLayoutPanel t = new TableLayoutPanel();
            t.Dock = DockStyle.Fill;
            t.Margin = new Padding(0);
            t.ColumnCount = 2;
            t.RowCount = 1;
            t.ColumnStyles.Add(new ColumnStyle(SizeType.AutoSize));
            t.ColumnStyles.Add(new ColumnStyle(SizeType.Percent, 100));

            FlowLayoutPanel primary = new FlowLayoutPanel();
            primary.AutoSize = true;
            primary.WrapContents = false;
            primary.Margin = new Padding(0);
            primary.Anchor = AnchorStyles.Left;

            openButton = MakeButton("打开工作台", ButtonKind.Primary, S(132));
            openButton.Click += delegate { OpenWorkbench(); };
            setupButton = MakeButton("初始化环境", ButtonKind.Secondary, S(120));
            setupButton.Click += delegate { StartSetup(); };
            primary.Controls.Add(openButton);
            primary.Controls.Add(setupButton);
            t.Controls.Add(primary, 0, 0);

            FlowLayoutPanel tools = new FlowLayoutPanel();
            tools.AutoSize = true;
            tools.WrapContents = false;
            tools.FlowDirection = FlowDirection.RightToLeft;
            tools.Margin = new Padding(0);
            tools.Anchor = AnchorStyles.Right;

            shortcutButton = MakeButton("修复桌面图标", ButtonKind.Quiet, 0);
            shortcutButton.Click += delegate { RepairShortcut(); };
            helpButton = MakeButton("使用说明", ButtonKind.Quiet, 0);
            helpButton.Click += delegate { OpenHelp(); };
            refreshButton = MakeButton("刷新检查", ButtonKind.Quiet, 0);
            refreshButton.Click += delegate { RefreshAll(); };
            stopButton = MakeButton("停止服务", ButtonKind.Quiet, 0);
            stopButton.Click += delegate { StopService(); };
            // RightToLeft：先加入的在最右侧。
            tools.Controls.Add(shortcutButton);
            tools.Controls.Add(helpButton);
            tools.Controls.Add(refreshButton);
            tools.Controls.Add(stopButton);
            t.Controls.Add(tools, 1, 0);
            return t;
        }

        enum ButtonKind { Primary, Secondary, Quiet }

        Button MakeButton(string text, ButtonKind kind, int width)
        {
            Button b = new Button();
            b.Text = text;
            b.FlatStyle = FlatStyle.Flat;
            b.UseVisualStyleBackColor = false;
            b.Cursor = Cursors.Hand;
            b.Height = S(40);
            b.Margin = new Padding(0, 0, S(10), 0);
            b.TextAlign = ContentAlignment.MiddleCenter;
            if (kind == ButtonKind.Quiet)
            {
                b.AutoSize = true;
                b.AutoSizeMode = AutoSizeMode.GrowAndShrink;
                b.Padding = new Padding(S(8), S(6), S(8), S(6));
                b.MinimumSize = new Size(0, S(36));
                b.Margin = new Padding(S(4), 0, 0, 0);
                b.Font = F(9f, FontStyle.Regular);
                b.BackColor = Palette.Page;
                b.ForeColor = Palette.Body;
                b.FlatAppearance.BorderSize = 0;
                b.FlatAppearance.MouseOverBackColor = Palette.Selected;
                b.FlatAppearance.MouseDownBackColor = Palette.Line;
            }
            else
            {
                b.Width = width;
                b.Font = F(10f, kind == ButtonKind.Primary ? FontStyle.Bold : FontStyle.Regular);
                b.FlatAppearance.BorderSize = 1;
                if (kind == ButtonKind.Primary)
                {
                    b.BackColor = Palette.Ink;
                    b.ForeColor = Color.White;
                    b.FlatAppearance.BorderColor = Palette.Ink;
                    b.FlatAppearance.MouseOverBackColor = Color.FromArgb(0x34, 0x3C, 0x45);
                    b.FlatAppearance.MouseDownBackColor = Color.FromArgb(0x0F, 0x14, 0x19);
                }
                else
                {
                    b.BackColor = Palette.Card;
                    b.ForeColor = Palette.Ink;
                    b.FlatAppearance.BorderColor = Palette.FieldLine;
                    b.FlatAppearance.MouseOverBackColor = Palette.Page;
                    b.FlatAppearance.MouseDownBackColor = Palette.Selected;
                }
            }
            return b;
        }

        Control BuildNotice()
        {
            TableLayoutPanel t = new TableLayoutPanel();
            t.Dock = DockStyle.Fill;
            t.Margin = new Padding(0, S(8), 0, 0);
            t.ColumnCount = 1;
            t.RowCount = 2;
            t.RowStyles.Add(new RowStyle(SizeType.Percent, 100));
            t.RowStyles.Add(new RowStyle(SizeType.Absolute, S(26)));

            noticeLabel = new Label();
            noticeLabel.Dock = DockStyle.Fill;
            noticeLabel.Margin = new Padding(0);
            noticeLabel.Font = F(9.5f, FontStyle.Regular);
            noticeLabel.ForeColor = Palette.Body;
            noticeLabel.AutoEllipsis = true;
            t.Controls.Add(noticeLabel, 0, 0);

            FlowLayoutPanel links = new FlowLayoutPanel();
            links.Dock = DockStyle.Fill;
            links.Margin = new Padding(0);
            links.WrapContents = false;
            pythonLink = MakeLink("Python 3.12 下载页", "https://www.python.org/downloads/windows/");
            nodeLink = MakeLink("Node.js 24 下载页", "https://nodejs.org/en/download");
            links.Controls.Add(pythonLink);
            links.Controls.Add(nodeLink);
            t.Controls.Add(links, 0, 1);
            return t;
        }

        LinkLabel MakeLink(string text, string url)
        {
            LinkLabel l = new LinkLabel();
            l.Text = text;
            l.AutoSize = true;
            l.Visible = false;
            l.Font = F(9f, FontStyle.Regular);
            l.LinkColor = Palette.Ink;
            l.ActiveLinkColor = Palette.Body;
            l.VisitedLinkColor = Palette.Ink;
            l.LinkBehavior = LinkBehavior.HoverUnderline;
            l.Margin = new Padding(0, 0, S(16), 0);
            l.LinkClicked += delegate { ShellOpen(url); };   // 只打开官方下载页，由用户自行安装
            return l;
        }

        Control BuildFooter()
        {
            TableLayoutPanel t = new TableLayoutPanel();
            t.Dock = DockStyle.Fill;
            t.Margin = new Padding(0);
            t.ColumnCount = 1;
            t.RowCount = 3;
            t.RowStyles.Add(new RowStyle(SizeType.Absolute, 1));
            t.RowStyles.Add(new RowStyle(SizeType.Percent, 60));
            t.RowStyles.Add(new RowStyle(SizeType.Percent, 40));

            Panel rule = new Panel();
            rule.BackColor = Palette.Line;
            rule.Dock = DockStyle.Fill;
            rule.Margin = new Padding(0);
            t.Controls.Add(rule, 0, 0);

            Label note = new Label();
            note.Text = "首次初始化可能下载数 GB 的依赖和模型，请留出磁盘空间和时间。提问时会调用你在工作台设置中配置的模型接口，费用由该服务商计算；启动器本身不调用模型。";
            note.Dock = DockStyle.Fill;
            note.Margin = new Padding(0, S(8), 0, 0);
            note.Font = F(8.5f, FontStyle.Regular);
            note.ForeColor = Palette.Sub;
            note.AutoEllipsis = true;
            t.Controls.Add(note, 0, 1);

            Label meta = new Label();
            meta.Text = "启动器 " + AppInfo.Version + "  ·  模型与资料保存在本机";
            meta.Dock = DockStyle.Fill;
            meta.Margin = new Padding(0);
            meta.Font = F(8f, FontStyle.Regular);
            meta.ForeColor = Palette.Muted;
            meta.AutoEllipsis = true;
            t.Controls.Add(meta, 0, 2);
            return t;
        }

        // ---------------- 生命周期

        protected override void OnShown(EventArgs e)
        {
            base.OnShown(e);
            UpdateUi();
            if (!noShortcut) AutoShortcut();
            RefreshAll();
            healthTimer.Start();
        }

        protected override void OnFormClosing(FormClosingEventArgs e)
        {
            if (setupRunner != null && !closeConfirmed && e.CloseReason == CloseReason.UserClosing)
            {
                DialogResult r = MessageBox.Show(this,
                    "初始化还在进行。关闭启动器会中止它，已下载的内容可能需要下次重新校验。\n\n确定要中止并关闭吗？",
                    "资料问答工作台", MessageBoxButtons.YesNo, MessageBoxIcon.Warning, MessageBoxDefaultButton.Button2);
                if (r != DialogResult.Yes)
                {
                    e.Cancel = true;
                    return;
                }
                closeConfirmed = true;
                SetupLog.Append(root, "[launcher] 用户关闭启动器，中止初始化。");
                setupRunner.Cancel();   // 只终止本启动器创建的 Job 内进程
            }
            healthTimer.Stop();
            base.OnFormClosing(e);
        }

        void OnUi(Action a)
        {
            if (IsDisposed || !IsHandleCreated) return;
            try { BeginInvoke(a); }
            catch (InvalidOperationException) { }
        }

        // ---------------- 状态渲染

        void SetCard(StatusDot dot, Label value, Label detail, Tone tone, string v, string d)
        {
            dot.DotColor = tone == Tone.Ok ? Palette.Ok : tone == Tone.Warn ? Palette.Warn : tone == Tone.Err ? Palette.Err : Palette.Dot;
            value.Text = v;
            value.ForeColor = tone == Tone.Err ? Palette.Err : Palette.Ink;
            detail.Text = d;
        }

        void Notice(string text, Tone tone)
        {
            noticeLabel.Text = text ?? "";
            noticeLabel.ForeColor = tone == Tone.Err ? Palette.Err : tone == Tone.Warn ? Palette.Warn : tone == Tone.Ok ? Palette.Ok : Palette.Body;
        }

        void UpdateUi()
        {
            // 环境与资料卡
            if (operation == "check" && lastCheck == null && checkProblem == null)
            {
                SetCard(envDot, envValue, envDetail, Tone.Neutral, "检查中…", "正在确认 Python、Node.js 和依赖。");
                SetCard(dataDot, dataValue, dataDetail, Tone.Neutral, "检查中…", "正在确认模型和索引。");
            }
            else if (checkProblem != null)
            {
                SetCard(envDot, envValue, envDetail, Tone.Warn, checkPythonMissing ? "未找到 Python 3.12" : "无法检查", checkProblem);
                SetCard(dataDot, dataValue, dataDetail, Tone.Neutral, "未知", "环境检查完成后显示。");
            }
            else if (lastCheck != null)
            {
                List<string> missing = new List<string>();
                if (!lastCheck.PythonReady) missing.Add("Python 3.12");
                if (!lastCheck.NodeReady) missing.Add("Node.js 24");
                if (!lastCheck.RequirementsReady) missing.Add("Python 依赖");
                if (missing.Count == 0)
                    SetCard(envDot, envValue, envDetail, Tone.Ok, "已就绪", "Python、Node.js 与依赖已安装。");
                else
                    SetCard(envDot, envValue, envDetail, Tone.Warn, "需要准备", "缺少：" + string.Join("、", missing.ToArray()));

                List<string> dataMissing = new List<string>();
                if (!lastCheck.ModelsReady) dataMissing.Add("模型");
                if (!lastCheck.IndexReady) dataMissing.Add("资料索引");
                if (dataMissing.Count == 0)
                    SetCard(dataDot, dataValue, dataDetail, Tone.Ok, "已就绪", "检索模型和资料索引可用。");
                else
                    SetCard(dataDot, dataValue, dataDetail, Tone.Warn, "需要初始化", "缺少：" + string.Join("、", dataMissing.ToArray()));
            }

            // 服务卡
            switch (lastHealth.State)
            {
                case ServiceState.Ours:
                    if (lastHealth.Busy)
                        SetCard(svcDot, svcValue, svcDetail, Tone.Ok, "运行中 · 正在回答", "有问题正在处理，完成前不能停止服务。");
                    else
                        SetCard(svcDot, svcValue, svcDetail, Tone.Ok, "运行中", lastHealth.Ready ? "可以打开工作台提问。" : SetupOutput.Shorten(lastHealth.Message, 60));
                    break;
                case ServiceState.OtherWorkspace:
                    SetCard(svcDot, svcValue, svcDetail, Tone.Warn, "端口被占用", "8765 端口上运行的是另一个目录的工作台。");
                    break;
                case ServiceState.Occupied:
                    SetCard(svcDot, svcValue, svcDetail, Tone.Warn, "端口被占用", SetupOutput.Shorten(lastHealth.Message, 60));
                    break;
                default:
                    SetCard(svcDot, svcValue, svcDetail, Tone.Neutral, "未启动", "点击“打开工作台”时启动。");
                    break;
            }

            bool idle = operation == null;
            bool ours = lastHealth.State == ServiceState.Ours;
            bool portFree = lastHealth.State == ServiceState.NotRunning;
            bool envReady = lastCheck != null && lastCheck.Ready;
            openButton.Enabled = idle && (ours || (portFree && envReady));
            setupButton.Enabled = idle && !ours && File.Exists(ScriptPath("_install_workbench.py"));
            stopButton.Enabled = idle && ours;
            refreshButton.Enabled = idle;
            shortcutButton.Enabled = idle;
            helpButton.Enabled = true;

            bool needPython = checkPythonMissing || (lastCheck != null && !lastCheck.PythonReady);
            bool needNode = lastCheck != null && !lastCheck.NodeReady;
            pythonLink.Visible = needPython;
            nodeLink.Visible = needNode;

            strip.Active = operation != null;
        }

        void Begin(string op, string label)
        {
            operation = op;
            stripLabel.Text = label;
            stripLabel.ForeColor = Palette.Body;
            UpdateUi();
        }

        void End()
        {
            operation = null;
            stripLabel.Text = idleHint ?? "";
            stripLabel.ForeColor = Palette.Muted;
            UpdateUi();
        }

        string ScriptPath(string name) { return Path.Combine(root, "scripts", name); }

        // ---------------- 检查

        void RefreshAll()
        {
            RefreshAll(false);
        }

        // keepNotice：初始化失败后刷新卡片，但保留失败说明，不被检查结果覆盖。
        void RefreshAll(bool keepNotice)
        {
            if (operation != null) return;
            if (!File.Exists(ScriptPath("_install_workbench.py")))
            {
                lastCheck = null;
                checkPythonMissing = false;
                checkProblem = "未找到 scripts\\_install_workbench.py，项目文件可能不完整。";
                ProbeHealth(null);
                UpdateUi();
                return;
            }

            Begin("check", "正在检查运行环境…");
            lastCheck = null;
            checkProblem = null;
            checkPythonMissing = false;
            PythonCommand py = PythonCommand.Resolve(root);
            ProcessRunner r = new ProcessRunner(py.FileName, py.With(ScriptPath("_install_workbench.py"), "--check"), root, false, 120000);
            r.Completed += delegate(ProcessOutcome o)
            {
                HealthInfo h = HealthInfo.Probe(workspaceId, 1500);
                OnUi(delegate
                {
                    lastHealth = h;
                    string keptText = noticeLabel.Text;
                    Color keptColor = noticeLabel.ForeColor;
                    ApplyCheck(py, o);
                    if (keepNotice)
                    {
                        noticeLabel.Text = keptText;
                        noticeLabel.ForeColor = keptColor;
                    }
                    End();
                });
            };
            r.Start();
        }

        void ApplyCheck(PythonCommand py, ProcessOutcome o)
        {
            if (o.StartFailed)
            {
                checkPythonMissing = py.UsesPyLauncher;
                checkProblem = py.UsesPyLauncher
                    ? "没有项目环境，也找不到 py 启动器。请先安装 Python 3.12（勾选 py launcher），再点“刷新检查”。"
                    : "无法运行项目 Python：" + SetupOutput.Shorten(o.StartError, 80);
                Notice(checkProblem, Tone.Warn);
                return;
            }
            CheckInfo c = CheckInfo.Parse(o.Stdout);
            if (c == null)
            {
                checkPythonMissing = py.UsesPyLauncher && o.ExitCode != 0;
                checkProblem = checkPythonMissing
                    ? "py 启动器找不到 Python 3.12。请安装 Python 3.12 后点“刷新检查”。"
                    : "环境检查没有返回结果" + (o.TimedOut ? "（超时）" : "") + "：" + SetupOutput.Shorten(o.LastErrorLine, 80);
                Notice(checkProblem, Tone.Warn);
                return;
            }
            lastCheck = c;
            if (c.Ready)
                Notice(lastHealth.State == ServiceState.Ours ? "工作台已在运行。" : "一切就绪，可以打开工作台。", Tone.Neutral);
            else
                Notice(c.Message.Length > 0 ? c.Message : "还有项目未准备好，点击“初始化环境”。", Tone.Warn);
            NoticeOccupied();
        }

        void NoticeOccupied()
        {
            if (lastHealth.State == ServiceState.OtherWorkspace)
                Notice("8765 端口正被另一个目录的工作台使用。请先在那个目录的启动器里停止它，本启动器不会接管或停止它。", Tone.Warn);
            else if (lastHealth.State == ServiceState.Occupied)
                Notice("8765 端口被其他程序占用，暂时无法启动工作台。关闭占用端口的程序后点“刷新检查”。", Tone.Warn);
        }

        void ProbeHealth(Action<HealthInfo> then)
        {
            if (Interlocked.CompareExchange(ref probing, 1, 0) != 0) return;
            ThreadPool.QueueUserWorkItem(delegate
            {
                HealthInfo h = HealthInfo.Probe(workspaceId, 1500);
                Interlocked.Exchange(ref probing, 0);
                OnUi(delegate
                {
                    lastHealth = h;
                    UpdateUi();
                    if (then != null) then(h);
                });
            });
        }

        // ---------------- 打开

        void OpenWorkbench()
        {
            if (operation != null) return;
            Begin("open", "正在确认服务状态…");
            ThreadPool.QueueUserWorkItem(delegate
            {
                HealthInfo h = HealthInfo.Probe(workspaceId, 1500);
                OnUi(delegate
                {
                    lastHealth = h;
                    if (h.State == ServiceState.Ours)
                    {
                        End();
                        OpenBrowser();
                        return;
                    }
                    if (h.State != ServiceState.NotRunning)
                    {
                        End();
                        NoticeOccupied();
                        return;
                    }
                    if (lastCheck == null || !lastCheck.Ready)
                    {
                        End();
                        Notice("环境还没有准备好，请先点击“初始化环境”。", Tone.Warn);
                        return;
                    }
                    LaunchBackend();
                });
            });
        }

        void LaunchBackend()
        {
            stripLabel.Text = "正在启动本机服务…";
            PythonCommand py = PythonCommand.Resolve(root);
            // 不放入 Job：服务需要在启动器关闭后继续运行。
            ProcessRunner r = new ProcessRunner(py.FileName, py.With(ScriptPath("_launch_workbench.py"), "--no-open"), root, false, 90000);
            r.Completed += delegate(ProcessOutcome o)
            {
                HealthInfo h = HealthInfo.NotRunning();
                if (!o.StartFailed && o.ExitCode == 0)
                {
                    for (int i = 0; i < 20; i++)
                    {
                        h = HealthInfo.Probe(workspaceId, 1500);
                        if (h.State != ServiceState.NotRunning) break;
                        Thread.Sleep(500);
                    }
                }
                OnUi(delegate
                {
                    lastHealth = h;
                    End();
                    if (h.State == ServiceState.Ours)
                    {
                        Notice("工作台已启动。", Tone.Ok);
                        OpenBrowser();      // 只有确认是本项目的服务后才打开网址
                    }
                    else if (h.State == ServiceState.OtherWorkspace || h.State == ServiceState.Occupied)
                    {
                        NoticeOccupied();
                    }
                    else
                    {
                        string why = o.StartFailed ? SetupOutput.Shorten(o.StartError, 80)
                                   : o.TimedOut ? "启动超时"
                                   : SetupOutput.Shorten(LastMeaningfulLine(o), 100);
                        Notice("服务没有启动成功" + (why.Length > 0 ? "：" + why : "") + "。详细记录见 data\\ui\\server.log。", Tone.Err);
                    }
                });
            };
            r.Start();
        }

        static string LastMeaningfulLine(ProcessOutcome o)
        {
            if (o.LastErrorLine.Trim().Length > 0) return o.LastErrorLine;
            string[] lines = o.Stdout.Split(new char[] { '\r', '\n' }, StringSplitOptions.RemoveEmptyEntries);
            return lines.Length > 0 ? lines[lines.Length - 1] : "";
        }

        void OpenBrowser()
        {
            if (!ShellOpen(AppInfo.BaseUrl))
                Notice("无法打开默认浏览器，请手动访问 " + AppInfo.BaseUrl, Tone.Warn);
        }

        static bool ShellOpen(string target)
        {
            try
            {
                ProcessStartInfo psi = new ProcessStartInfo(target);
                psi.UseShellExecute = true;
                Process p = Process.Start(psi);
                if (p != null) p.Dispose();
                return true;
            }
            catch (Exception)
            {
                return false;
            }
        }

        // ---------------- 停止

        void StopService()
        {
            if (operation != null) return;
            Begin("stop", "正在确认服务状态…");
            ThreadPool.QueueUserWorkItem(delegate
            {
                HealthInfo h = HealthInfo.Probe(workspaceId, 1500);
                OnUi(delegate
                {
                    lastHealth = h;
                    if (h.State != ServiceState.Ours)
                    {
                        End();
                        if (h.State == ServiceState.NotRunning) Notice("服务没有在运行。", Tone.Neutral);
                        else Notice("端口上运行的不是本项目的工作台，不会停止它。", Tone.Warn);
                        return;
                    }
                    if (h.Busy)
                    {
                        End();
                        Notice("有问题正在回答，现在停止会丢失这次结果。请等回答完成后再停止。", Tone.Warn);
                        return;
                    }
                    stripLabel.Text = "正在停止服务…";
                    PythonCommand py = PythonCommand.Resolve(root);
                    ProcessRunner r = new ProcessRunner(py.FileName, py.With(ScriptPath("_launch_workbench.py"), "--stop"), root, false, 30000);
                    r.Completed += delegate(ProcessOutcome o)
                    {
                        HealthInfo after = HealthInfo.Probe(workspaceId, 1500);
                        for (int i = 0; i < 20 && after.State == ServiceState.Ours; i++)
                        {
                            Thread.Sleep(500);
                            after = HealthInfo.Probe(workspaceId, 1500);
                        }
                        OnUi(delegate
                        {
                            lastHealth = after;
                            End();
                            if (after.State == ServiceState.Ours)
                                Notice("服务还没有停止" + (o.ExitCode != 0 ? "：" + SetupOutput.Shorten(LastMeaningfulLine(o), 80) : "") + "。", Tone.Err);
                            else
                                Notice("服务已停止。", Tone.Neutral);
                        });
                    };
                    r.Start();
                });
            });
        }

        // ---------------- 初始化

        void StartSetup()
        {
            if (operation != null) return;
            if (!File.Exists(ScriptPath("_install_workbench.py")))
            {
                Notice("未找到 scripts\\_install_workbench.py，无法初始化。", Tone.Err);
                return;
            }
            if (lastHealth.State == ServiceState.Ours)
            {
                Notice("工作台正在运行。请先停止服务，再初始化环境。", Tone.Warn);
                return;
            }
            DialogResult confirm = MessageBox.Show(this,
                "初始化会安装固定版本的依赖、下载检索模型并准备资料索引，可能下载数 GB，耗时较长。\n" +
                "过程中不会调用付费模型接口。\n\n现在开始吗？",
                "初始化环境", MessageBoxButtons.OKCancel, MessageBoxIcon.Information, MessageBoxDefaultButton.Button1);
            if (confirm != DialogResult.OK) return;

            Begin("setup", "正在初始化…");
            Notice("初始化进行中，可以最小化启动器。完整记录写入 data\\ui\\setup.log。", Tone.Neutral);
            SetupLog.Append(root, "");
            SetupLog.Append(root, "===== 初始化开始 " + DateTime.Now.ToString("yyyy-MM-dd HH:mm:ss") + " =====");

            PythonCommand py = PythonCommand.Resolve(root);
            ProcessRunner r = new ProcessRunner(py.FileName, py.With(ScriptPath("_install_workbench.py"), "--install"), root, true, 0);
            setupRunner = r;
            string lastProblem = "";
            r.Line += delegate(string text, bool isError)
            {
                string safe = SetupOutput.Redact(text);
                SetupLog.Append(root, (isError ? "[stderr] " : "") + safe);
                StageLine stage = SetupOutput.ParseStage(safe);
                if (stage != null)
                {
                    OnUi(delegate { if (operation == "setup") stripLabel.Text = stage.Message.Length > 0 ? stage.Message : stage.Key; });
                }
                else if (safe.Trim().Length > 0 && (isError || Regex.IsMatch(safe, "error|failed|失败|错误", RegexOptions.IgnoreCase)))
                {
                    lastProblem = safe;
                }
            };
            r.Completed += delegate(ProcessOutcome o)
            {
                SetupLog.Append(root, "===== 初始化结束，退出码 " + o.ExitCode + (o.Cancelled ? "（已中止）" : "") + " =====");
                OnUi(delegate
                {
                    setupRunner = null;
                    End();
                    if (closeConfirmed) return;
                    if (o.StartFailed)
                    {
                        checkPythonMissing = py.UsesPyLauncher;
                        Notice(py.UsesPyLauncher ? "找不到 py 启动器。请先安装 Python 3.12，然后重试。" : "无法运行项目 Python：" + SetupOutput.Shorten(o.StartError, 80), Tone.Err);
                        UpdateUi();
                        return;
                    }
                    if (o.ExitCode == 0)
                    {
                        Notice("初始化完成。", Tone.Ok);
                        RefreshAll();   // 检查结果会接着更新提示，例如“一切就绪”
                    }
                    else if (o.Cancelled)
                    {
                        Notice("初始化已中止。完整记录见 data\\ui\\setup.log。", Tone.Warn);
                        RefreshAll(true);
                    }
                    else
                    {
                        string detail = SetupOutput.Shorten(lastProblem.Length > 0 ? lastProblem : o.LastErrorLine, 110);
                        Notice("初始化未完成" + (detail.Length > 0 ? "：" + detail : "") + "。完整记录见 data\\ui\\setup.log。", Tone.Err);
                        RefreshAll(true);
                    }
                });
            };
            r.Start();
        }

        // ---------------- 其他

        void OpenHelp()
        {
            string[] candidates = new string[]
            {
                Path.Combine(root, @"doc\UI-本地工作台接入.md"),
                Path.Combine(root, "README.md"),
            };
            foreach (string c in candidates)
            {
                if (File.Exists(c) && ShellOpen(c)) return;
            }
            Notice("没有找到使用说明文件，请查看项目目录中的 README.md。", Tone.Warn);
        }

        void AutoShortcut()
        {
            string state = ShortcutState.StatePath(root);
            if (!ShortcutState.ShouldAutoEnsure(state, root)) return;
            ShortcutResult result = EnsureDesktopShortcut();
            if (result.Succeeded)
            {
                try { ShortcutState.MarkDone(state, root, result); }
                catch (Exception) { }
            }
            // 失败不弹窗、不影响其余功能；结果作为轻提示显示，可用“修复桌面图标”重试。
            if (result.Status != ShortcutStatus.Unchanged)
                idleHint = result.Message + (result.Succeeded ? "" : " 可点“修复桌面图标”重试。");
        }

        void RepairShortcut()
        {
            ShortcutResult result = EnsureDesktopShortcut();
            idleHint = null;
            stripLabel.Text = "";
            if (result.Succeeded)
            {
                try { ShortcutState.MarkDone(ShortcutState.StatePath(root), root, result); }
                catch (Exception) { }
                Notice(result.Message + "  " + Path.GetFileName(result.Path), Tone.Ok);
            }
            else
            {
                Notice(result.Message, result.Status == ShortcutStatus.Conflict ? Tone.Warn : Tone.Err);
            }
        }

        ShortcutResult EnsureDesktopShortcut()
        {
            try
            {
                return ShortcutManager.Ensure(root, ShortcutManager.DesktopDirectory());
            }
            catch (Exception ex)
            {
                return ShortcutResult.Make(ShortcutStatus.Failed, "", "无法创建桌面图标：" + SetupOutput.Shorten(ex.Message, 100));
            }
        }

        protected override void Dispose(bool disposing)
        {
            if (disposing) healthTimer.Dispose();
            base.Dispose(disposing);
        }
    }

    public static class SetupLog
    {
        static readonly object Gate = new object();

        public static string PathFor(string root) { return Path.Combine(root, @"data\ui\setup.log"); }

        public static void Append(string root, string line)
        {
            try
            {
                lock (Gate)
                {
                    string path = PathFor(root);
                    Directory.CreateDirectory(Path.GetDirectoryName(path));
                    File.AppendAllText(path, (line ?? "") + Environment.NewLine, new UTF8Encoding(false));
                }
            }
            catch (Exception) { }   // 写日志失败不应中断安装
        }
    }

    // ------------------------------------------------------------------ 入口

    public static class Program
    {
        [DllImport("user32.dll")]
        static extern bool SetProcessDPIAware();

        [DllImport("kernel32.dll")]
        static extern bool AttachConsole(int processId);

        public static string DefaultRoot()
        {
            return Workspace.NormalizeRoot(Path.GetDirectoryName(Application.ExecutablePath));
        }

        // 无界面的快捷方式命令。返回退出码，供测试直接调用。
        public static int RunInstallShortcut(LauncherOptions options, string root, TextWriter output)
        {
            string dir = options.ShortcutDir ?? ShortcutManager.DesktopDirectory();
            ShortcutResult result = ShortcutManager.Ensure(root, dir);
            if (output != null)
                output.WriteLine("shortcut " + result.Status.ToString().ToLowerInvariant() + ": " + result.Path);
            if (result.Succeeded && options.ShortcutDir == null)
            {
                try { ShortcutState.MarkDone(ShortcutState.StatePath(root), root, result); }
                catch (Exception) { }
            }
            return result.ExitCode;
        }

        [STAThread]
        public static int Main(string[] args)
        {
            LauncherOptions options = LauncherOptions.Parse(args);
            string root = DefaultRoot();

            if (!options.IsValid || options.InstallShortcut)
            {
                // winexe 默认没有控制台；从终端调用时附着到父控制台输出一行结果。
                TextWriter output = null;
                if (AttachConsole(-1))
                {
                    try
                    {
                        output = new StreamWriter(Console.OpenStandardOutput());
                        ((StreamWriter)output).AutoFlush = true;
                        output.WriteLine();
                    }
                    catch (Exception) { output = null; }
                }
                if (!options.IsValid)
                {
                    if (output != null) output.WriteLine(options.Error + "\n" + LauncherOptions.Usage);
                    return ExitCodes.Usage;
                }
                try
                {
                    return RunInstallShortcut(options, root, output);
                }
                catch (Exception ex)
                {
                    if (output != null) output.WriteLine("shortcut failed: " + ex.Message);
                    return ExitCodes.ShortcutFailed;
                }
            }

            try { SetProcessDPIAware(); }
            catch (Exception) { }
            Application.EnableVisualStyles();
            Application.SetCompatibleTextRenderingDefault(false);

            bool created;
            using (Mutex single = new Mutex(true, @"Local\RagWorkbenchLauncher-" + Workspace.ComputeId(root), out created))
            {
                if (!created)
                {
                    MessageBox.Show("这个目录的启动器已经打开。", "资料问答工作台", MessageBoxButtons.OK, MessageBoxIcon.Information);
                    return ExitCodes.AlreadyRunning;
                }
                Application.Run(new LauncherForm(root, options.NoShortcut));
                return ExitCodes.Ok;
            }
        }
    }
}
