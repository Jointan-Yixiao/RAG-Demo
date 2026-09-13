// 启动器离线测试。与 launcher/WorkbenchLauncher.cs 一起编译为控制台程序：
//   csc /target:exe /main:RagWorkbench.Tests.LauncherTests ... launcher\WorkbenchLauncher.cs tests\launcher\LauncherTests.cs
// 用法：LauncherTests.exe <工作目录>
//   工作目录下为每次运行新建 run-<时间戳>，所有临时根目录与快捷方式都放在其中；测试不删除文件。
// 不访问网络、不调用模型、不写真实桌面。

using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading;

namespace RagWorkbench.Tests
{
    public static class LauncherTests
    {
        static int passed, failed;
        static string runDir;
        static string selfPath;

        [STAThread]   // WScript.Shell 为单线程单元 COM 组件，与界面线程保持一致
        public static int Main(string[] args)
        {
            // 子进程模式：供进程相关测试调用自身。
            if (args.Length > 0 && args[0] == "--echo-args") return EchoArgs(args);
            if (args.Length > 0 && args[0] == "--sleep") { Thread.Sleep(int.Parse(args[1])); return 0; }
            if (args.Length > 0 && args[0] == "--spawn-sleeper") return SpawnSleeper();

            selfPath = System.Reflection.Assembly.GetExecutingAssembly().Location;
            string work = args.Length > 0 ? args[0] : Path.Combine(Environment.CurrentDirectory, "launcher-test-work");
            runDir = Path.Combine(Path.GetFullPath(work), "run-" + DateTime.Now.ToString("yyyyMMdd-HHmmss-fff"));
            Directory.CreateDirectory(runDir);
            try { Console.OutputEncoding = new UTF8Encoding(false); }
            catch (IOException) { }
            Console.WriteLine("work: " + runDir);

            Run("quote: plain, empty, spaces", QuotePlain);
            Run("quote: backslashes and quotes", QuoteBackslashes);
            Run("quote: round-trip through CommandLineToArgvW", QuoteRoundTrip);
            Run("options: parse valid combinations", OptionsValid);
            Run("options: reject invalid combinations", OptionsInvalid);
            Run("workspace: id matches backend contract vectors", WorkspaceVectors);
            Run("workspace: trailing separator and format", WorkspaceNormalize);
            Run("health: same workspace is ours", HealthOurs);
            Run("health: other root / old service / other program", HealthNotOurs);
            Run("check: parse JSON output", CheckParse);
            Run("setup: STAGE line parsing and redaction", StageParse);
            Run("shortcut: create, persist, idempotent", ShortcutCreateIdempotent);
            Run("shortcut: repair changed settings in place", ShortcutRepair);
            Run("shortcut: foreign live link is not overwritten", ShortcutCollision);
            Run("shortcut: both names foreign -> conflict", ShortcutConflict);
            Run("shortcut: stale link from moved project is repaired", ShortcutStale);
            Run("shortcut: no duplicate after primary name frees up", ShortcutNoDuplicate);
            Run("shortcut: CLI exit codes", ShortcutCli);
            Run("shortcut: auto-create state marker", ShortcutStateMarker);
            Run("process: hidden child receives exact UTF-8 args", ProcessArgs);
            Run("process: cancel terminates owned tree only", ProcessCancelTree);

            Console.WriteLine();
            Console.WriteLine("passed " + passed + ", failed " + failed);
            return failed == 0 ? 0 : 1;
        }

        // ------------------------------------------------------------ harness

        static void Run(string name, Action test)
        {
            try
            {
                test();
                passed++;
                Console.WriteLine("PASS  " + name);
            }
            catch (Exception ex)
            {
                failed++;
                Console.WriteLine("FAIL  " + name);
                Console.WriteLine("      " + ex.GetType().Name + ": " + ex.Message);
            }
        }

        static void Eq<T>(T expected, T actual, string what)
        {
            if (!EqualityComparer<T>.Default.Equals(expected, actual))
                throw new Exception(what + ": expected <" + expected + "> got <" + actual + ">");
        }

        static void True(bool cond, string what)
        {
            if (!cond) throw new Exception(what);
        }

        static string NewDir(string name)
        {
            string d = Path.Combine(runDir, name);
            Directory.CreateDirectory(d);
            return d;
        }

        // 伪造一个项目根目录：只需要 RAG-Workbench.exe 与 assets\app.ico 存在（内容无关）。
        static string FakeRoot(string parent, string folderName)
        {
            string root = Path.Combine(parent, folderName);
            Directory.CreateDirectory(Path.Combine(root, "assets"));
            File.WriteAllBytes(Path.Combine(root, "RAG-Workbench.exe"), new byte[0]);
            File.WriteAllBytes(Path.Combine(root, @"assets\app.ico"), new byte[0]);
            return root;
        }

        static int LinkCount(string dir)
        {
            return Directory.GetFiles(dir, "*.lnk").Length;
        }

        // ------------------------------------------------------------ quoting

        static void QuotePlain()
        {
            Eq("abc", WinArgs.Quote("abc"), "plain");
            Eq("\"\"", WinArgs.Quote(""), "empty");
            Eq("\"\"", WinArgs.Quote(null), "null");
            Eq("\"a b\"", WinArgs.Quote("a b"), "space");
            Eq("\"C:\\资料 目录\\x.py\"", WinArgs.Quote("C:\\资料 目录\\x.py"), "chinese with space");
            Eq("C:\\资料\\x.py", WinArgs.Quote("C:\\资料\\x.py"), "chinese without space stays bare");
            Eq("-3.12 -X utf8 \"C:\\my dir\\s.py\" --check",
               WinArgs.Join(new string[] { "-3.12", "-X", "utf8", "C:\\my dir\\s.py", "--check" }), "join");
        }

        static void QuoteBackslashes()
        {
            Eq("C:\\dir\\", WinArgs.Quote("C:\\dir\\"), "trailing backslash without space stays bare");
            Eq("\"C:\\my dir\\\\\"", WinArgs.Quote("C:\\my dir\\"), "trailing backslash doubled before closing quote");
            Eq("\"say \\\"hi\\\"\"", WinArgs.Quote("say \"hi\""), "embedded quotes");
            Eq("\"a\\\\\\\"b\"", WinArgs.Quote("a\\\"b"), "backslash before quote");
            Eq("\"a\\\\b c\"", WinArgs.Quote("a\\\\b c"), "backslashes not before quote kept");
        }

        [DllImport("shell32.dll", SetLastError = true, CharSet = CharSet.Unicode)]
        static extern IntPtr CommandLineToArgvW(string cmdLine, out int argc);

        [DllImport("kernel32.dll")]
        static extern IntPtr LocalFree(IntPtr mem);

        static string[] SplitCommandLine(string cmd)
        {
            int argc;
            IntPtr argv = CommandLineToArgvW(cmd, out argc);
            if (argv == IntPtr.Zero) throw new Exception("CommandLineToArgvW failed");
            try
            {
                string[] result = new string[argc];
                for (int i = 0; i < argc; i++)
                    result[i] = Marshal.PtrToStringUni(Marshal.ReadIntPtr(argv, i * IntPtr.Size));
                return result;
            }
            finally
            {
                LocalFree(argv);
            }
        }

        static readonly string[] Tricky = new string[]
        {
            "", " ", "plain", "two words", "C:\\Program Files\\x\\", "C:\\资料 问答\\RAG 项目",
            "quote\"inside", "\\\"", "ends with backslash\\", "\\\\server\\share\\a b", "tab\there", "中文",
            "a\\\\\"b", "--shortcut-dir", "C:\\Users\\测试\\Desktop\\",
        };

        static void QuoteRoundTrip()
        {
            List<string> all = new List<string>(Tricky);
            string cmd = "prog.exe " + WinArgs.Join(all);
            string[] parsed = SplitCommandLine(cmd);
            Eq(all.Count + 1, parsed.Length, "argc");
            for (int i = 0; i < all.Count; i++) Eq(all[i], parsed[i + 1], "arg " + i);
        }

        // ------------------------------------------------------------ options

        static void OptionsValid()
        {
            LauncherOptions o = LauncherOptions.Parse(new string[0]);
            True(o.IsValid && !o.InstallShortcut && !o.NoShortcut && o.ShortcutDir == null, "empty");

            o = LauncherOptions.Parse(new string[] { "--no-shortcut" });
            True(o.IsValid && o.NoShortcut, "no-shortcut");

            o = LauncherOptions.Parse(new string[] { "--install-shortcut" });
            True(o.IsValid && o.InstallShortcut && o.ShortcutDir == null, "install-shortcut");

            o = LauncherOptions.Parse(new string[] { "--install-shortcut", "--shortcut-dir", "C:\\临时 目录\\桌面" });
            True(o.IsValid, "install with dir: " + o.Error);
            Eq("C:\\临时 目录\\桌面", o.ShortcutDir, "dir value");

            o = LauncherOptions.Parse(new string[] { "--shortcut-dir", "\\\\server\\share\\d", "--install-shortcut" });
            True(o.IsValid, "order independent, UNC path: " + o.Error);
        }

        static void OptionsInvalid()
        {
            string[][] bad = new string[][]
            {
                new string[] { "--bogus" },
                new string[] { "--install-shortcut", "--shortcut-dir" },
                new string[] { "--install-shortcut", "--shortcut-dir", "--no-shortcut" },
                new string[] { "--install-shortcut", "--shortcut-dir", "relative\\dir" },
                new string[] { "--install-shortcut", "--shortcut-dir", "C:relative" },
                new string[] { "--install-shortcut", "--shortcut-dir", "\\rooted-no-drive" },
                new string[] { "--shortcut-dir", "C:\\x" },
                new string[] { "--install-shortcut", "--no-shortcut" },
                new string[] { "--install-shortcut", "--shortcut-dir", "C:\\a", "--shortcut-dir", "C:\\b" },
            };
            foreach (string[] a in bad)
            {
                LauncherOptions o = LauncherOptions.Parse(a);
                True(!o.IsValid, "should reject: " + string.Join(" ", a));
            }
        }

        // ------------------------------------------------------------ workspace

        static void WorkspaceVectors()
        {
            // 期望值由 sha256(utf8(path))[:16] 独立计算，与 Python 端 hashlib 结果一致。
            Eq("a929b865e65e5122", Workspace.ComputeId("C:\\Users\\测试\\RAG 项目"), "chinese path vector");
            Eq("5cbfb8a8772f9493", Workspace.ComputeId("C:\\Users\\Tan\\Desktop\\RAG-Demo"), "ascii path vector");
        }

        static void WorkspaceNormalize()
        {
            string a = Workspace.ComputeId("C:\\Users\\测试\\RAG 项目");
            Eq(a, Workspace.ComputeId("C:\\Users\\测试\\RAG 项目\\"), "trailing backslash");
            Eq(a, Workspace.ComputeId("C:\\Users\\测试\\RAG 项目/"), "trailing slash");
            True(a != Workspace.ComputeId("C:\\Users\\测试\\RAG 项目2"), "different root differs");
            Eq(a, Workspace.ComputeId("c:\\users\\测试\\rag 项目"), "case insensitive Windows path");
            Eq(16, a.Length, "length");
            foreach (char c in a) True((c >= '0' && c <= '9') || (c >= 'a' && c <= 'f'), "lowercase hex");
        }

        // ------------------------------------------------------------ health

        static void HealthOurs()
        {
            string id = Workspace.ComputeId("C:\\Users\\测试\\RAG 项目");
            HealthInfo h = HealthInfo.Classify(
                "{\"service\":\"rag-workbench\",\"ready\":true,\"busy\":true,\"message\":\"本地资料已就绪\",\"workspace_id\":\"" + id + "\"}", id);
            Eq(ServiceState.Ours, h.State, "state");
            True(h.Ready && h.Busy, "ready/busy parsed");
            Eq("本地资料已就绪", h.Message, "message");

            h = HealthInfo.Classify("{\"service\":\"rag-workbench\",\"ready\":false,\"busy\":false,\"workspace_id\":\"" + id.ToUpperInvariant() + "\"}", id);
            Eq(ServiceState.Ours, h.State, "hex case-insensitive");
            True(!h.Busy, "busy false");
        }

        static void HealthNotOurs()
        {
            string id = Workspace.ComputeId("C:\\Users\\测试\\RAG 项目");
            string other = Workspace.ComputeId("D:\\另一个\\RAG 项目");
            Eq(ServiceState.OtherWorkspace,
               HealthInfo.Classify("{\"service\":\"rag-workbench\",\"ready\":true,\"busy\":false,\"workspace_id\":\"" + other + "\"}", id).State,
               "other root");
            Eq(ServiceState.Occupied,
               HealthInfo.Classify("{\"service\":\"rag-workbench\",\"ready\":true,\"busy\":false}", id).State,
               "old service without workspace_id cannot prove identity");
            Eq(ServiceState.Occupied,
               HealthInfo.Classify("{\"service\":\"something-else\",\"workspace_id\":\"" + id + "\"}", id).State,
               "different service with our id");
            Eq(ServiceState.Occupied, HealthInfo.Classify("<html>hello</html>", id).State, "non-JSON");
            Eq(ServiceState.Occupied, HealthInfo.Classify("[1,2]", id).State, "JSON array");
            Eq(ServiceState.Occupied, HealthInfo.Classify("", id).State, "empty body");
            HealthInfo busyString = HealthInfo.Classify("{\"service\":\"rag-workbench\",\"busy\":\"true\",\"workspace_id\":\"" + id + "\"}", id);
            True(!busyString.Busy, "busy must be a real boolean");
        }

        // ------------------------------------------------------------ install backend output

        static void CheckParse()
        {
            string json = "{\"python_ready\":true,\"node_ready\":false,\"models_ready\":true,\"index_ready\":false," +
                          "\"requirements_ready\":true,\"ready\":false,\"message\":\"缺少 Node.js 24\"}";
            CheckInfo c = CheckInfo.Parse(json);
            True(c != null, "parsed");
            True(c.PythonReady && !c.NodeReady && c.ModelsReady && !c.IndexReady && c.RequirementsReady && !c.Ready, "flags");
            True(!c.EnvironmentReady && !c.DataReady, "derived");
            Eq("缺少 Node.js 24", c.Message, "message");

            CheckInfo noisy = CheckInfo.Parse("WARNING: something\r\n" + json + "\r\n");
            True(noisy != null && !noisy.NodeReady, "tolerates leading warning line");

            True(CheckInfo.Parse("") == null, "empty");
            True(CheckInfo.Parse("Traceback (most recent call last):") == null, "traceback");
            True(CheckInfo.Parse("{\"python_ready\":true}") == null, "missing ready is not a check result");
        }

        static void StageParse()
        {
            StageLine s = SetupOutput.ParseStage("STAGE|models|正在下载模型 | 2/3");
            True(s != null, "parsed");
            Eq("models", s.Key, "key");
            Eq("正在下载模型 | 2/3", s.Message, "message keeps pipes");

            s = SetupOutput.ParseStage("STAGE|done|");
            True(s != null && s.Key == "done" && s.Message == "", "empty message");

            True(SetupOutput.ParseStage("STAGE||x") == null, "empty key");
            True(SetupOutput.ParseStage("STAGE|nokey") == null, "missing second separator");
            True(SetupOutput.ParseStage("pip install ... STAGE|x|y") == null, "must start the line");
            True(SetupOutput.ParseStage(null) == null, "null");

            string red = SetupOutput.Redact("using key sk-abcdefghijklmnopqrstuvwx now");
            True(red.IndexOf("sk-abcdef") < 0, "sk- key redacted: " + red);
            red = SetupOutput.Redact("API_KEY=secret123 rest");
            True(red.IndexOf("secret123") < 0, "api_key= redacted: " + red);
            Eq("Collecting torch==2.7.0", SetupOutput.Redact("Collecting torch==2.7.0"), "ordinary line untouched");
        }

        // ------------------------------------------------------------ shortcuts

        static void ShortcutCreateIdempotent()
        {
            string box = NewDir("sc-create");
            string root = FakeRoot(box, "RAG 项目 甲");
            string desk = NewDir("sc-create\\desk");

            ShortcutResult r = ShortcutManager.Ensure(root, desk);
            Eq(ShortcutStatus.Created, r.Status, "first run: " + r.Message);
            Eq(Path.Combine(desk, "RAG 工作台.lnk"), r.Path, "primary name");
            True(File.Exists(r.Path), "file exists");

            // 用新的 COM 实例读回，确认已落盘。
            ShortcutInfo info = ShortcutManager.Read(r.Path);
            True(ShortcutManager.SamePath(info.TargetPath, Path.Combine(root, "RAG-Workbench.exe")), "target: " + info.TargetPath);
            True(ShortcutManager.SamePath(info.WorkingDirectory, root), "working dir: " + info.WorkingDirectory);
            True(info.IconLocation.StartsWith(Path.Combine(root, "assets\\app.ico"), StringComparison.OrdinalIgnoreCase), "icon: " + info.IconLocation);

            DateTime stamp = File.GetLastWriteTimeUtc(r.Path);
            Thread.Sleep(1100);
            ShortcutResult again = ShortcutManager.Ensure(root, desk);
            Eq(ShortcutStatus.Unchanged, again.Status, "second run");
            Eq(r.Path, again.Path, "same file");
            Eq(stamp, File.GetLastWriteTimeUtc(r.Path), "not rewritten");
            Eq(1, LinkCount(desk), "no duplicates");

            ShortcutResult third = ShortcutManager.Ensure(root + "\\", desk);
            Eq(ShortcutStatus.Unchanged, third.Status, "root with trailing slash is the same root");
        }

        static void ShortcutRepair()
        {
            string box = NewDir("sc-repair");
            string root = FakeRoot(box, "项目");
            string desk = NewDir("sc-repair\\desk");
            string path = Path.Combine(desk, "RAG 工作台.lnk");

            ShortcutManager.WriteLink(path, Path.Combine(root, "RAG-Workbench.exe"), box, "C:\\Windows\\System32\\shell32.dll,3");
            ShortcutResult r = ShortcutManager.Ensure(root, desk);
            Eq(ShortcutStatus.Repaired, r.Status, "repaired");
            Eq(path, r.Path, "in place");
            ShortcutInfo info = ShortcutManager.Read(path);
            True(ShortcutManager.SamePath(info.WorkingDirectory, root), "working dir fixed");
            True(info.IconLocation.StartsWith(Path.Combine(root, "assets\\app.ico"), StringComparison.OrdinalIgnoreCase), "icon fixed");
            Eq(ShortcutStatus.Unchanged, ShortcutManager.Ensure(root, desk).Status, "then stable");
            Eq(1, LinkCount(desk), "count");
        }

        static void ShortcutCollision()
        {
            string box = NewDir("sc-collide");
            string root = FakeRoot(box, "我的 RAG");
            string foreignExe = Path.Combine(NewDir("sc-collide\\other-app"), "other.exe");
            File.WriteAllBytes(foreignExe, new byte[0]);
            string desk = NewDir("sc-collide\\desk");
            string primary = Path.Combine(desk, "RAG 工作台.lnk");
            ShortcutManager.WriteLink(primary, foreignExe, Path.GetDirectoryName(foreignExe), foreignExe + ",0");
            DateTime stamp = File.GetLastWriteTimeUtc(primary);

            ShortcutResult r = ShortcutManager.Ensure(root, desk);
            Eq(ShortcutStatus.Created, r.Status, "alternate created: " + r.Message);
            Eq(Path.Combine(desk, "RAG 工作台 (我的 RAG).lnk"), r.Path, "alternate uses root folder name");
            True(ShortcutManager.SamePath(ShortcutManager.Read(primary).TargetPath, foreignExe), "foreign link untouched");
            Eq(stamp, File.GetLastWriteTimeUtc(primary), "foreign file not rewritten");

            Eq(ShortcutStatus.Unchanged, ShortcutManager.Ensure(root, desk).Status, "rerun stable");
            Eq(2, LinkCount(desk), "exactly two links");
        }

        static void ShortcutConflict()
        {
            string box = NewDir("sc-conflict");
            string root = FakeRoot(box, "RAG");
            string foreignExe = Path.Combine(NewDir("sc-conflict\\other"), "tool.exe");
            File.WriteAllBytes(foreignExe, new byte[0]);
            string desk = NewDir("sc-conflict\\desk");
            ShortcutManager.WriteLink(Path.Combine(desk, "RAG 工作台.lnk"), foreignExe, "", foreignExe + ",0");
            ShortcutManager.WriteLink(Path.Combine(desk, "RAG 工作台 (RAG).lnk"), foreignExe, "", foreignExe + ",0");

            ShortcutResult r = ShortcutManager.Ensure(root, desk);
            Eq(ShortcutStatus.Conflict, r.Status, "conflict");
            Eq(ExitCodes.ShortcutConflict, r.ExitCode, "exit code");
            Eq(2, LinkCount(desk), "nothing added");

            // 同名的非 .lnk 目录也不能被覆盖。
            string desk2 = NewDir("sc-conflict\\desk2");
            Directory.CreateDirectory(Path.Combine(desk2, "RAG 工作台.lnk"));
            ShortcutResult r2 = ShortcutManager.Ensure(root, desk2);
            Eq(ShortcutStatus.Created, r2.Status, "directory named like link is skipped");
            Eq(Path.Combine(desk2, "RAG 工作台 (RAG).lnk"), r2.Path, "alternate");
        }

        static void ShortcutStale()
        {
            string box = NewDir("sc-stale");
            string root = FakeRoot(box, "新位置");
            string desk = NewDir("sc-stale\\desk");
            string primary = Path.Combine(desk, "RAG 工作台.lnk");
            string movedAway = Path.Combine(box, "旧位置不存在\\RAG-Workbench.exe");
            ShortcutManager.WriteLink(primary, movedAway, Path.GetDirectoryName(movedAway), "C:\\Windows\\System32\\shell32.dll,0");

            ShortcutResult r = ShortcutManager.Ensure(root, desk);
            Eq(ShortcutStatus.Repaired, r.Status, "stale launcher link repaired");
            Eq(primary, r.Path, "in place");
            True(ShortcutManager.SamePath(ShortcutManager.Read(primary).TargetPath, Path.Combine(root, "RAG-Workbench.exe")), "retargeted");

            // 指向不存在的其他程序：不是我们的，不改。
            string desk2 = NewDir("sc-stale\\desk2");
            string primary2 = Path.Combine(desk2, "RAG 工作台.lnk");
            ShortcutManager.WriteLink(primary2, Path.Combine(box, "gone\\other.exe"), "", "C:\\Windows\\System32\\shell32.dll,0");
            ShortcutResult r2 = ShortcutManager.Ensure(root, desk2);
            Eq(ShortcutStatus.Created, r2.Status, "dead foreign link left alone");
            Eq(Path.Combine(desk2, "RAG 工作台 (新位置).lnk"), r2.Path, "alternate");
        }

        static void ShortcutNoDuplicate()
        {
            string box = NewDir("sc-nodup");
            string root = FakeRoot(box, "RAG");
            string foreignExe = Path.Combine(NewDir("sc-nodup\\other"), "tool.exe");
            File.WriteAllBytes(foreignExe, new byte[0]);
            string desk = NewDir("sc-nodup\\desk");
            string primary = Path.Combine(desk, "RAG 工作台.lnk");
            ShortcutManager.WriteLink(primary, foreignExe, "", foreignExe + ",0");
            ShortcutResult first = ShortcutManager.Ensure(root, desk);
            Eq(ShortcutStatus.Created, first.Status, "alternate created");

            // 用户把占用主名称的快捷方式挪走（改名，不删除），主名称空出来。
            File.Move(primary, Path.Combine(desk, "tool.lnk"));
            ShortcutResult second = ShortcutManager.Ensure(root, desk);
            Eq(ShortcutStatus.Unchanged, second.Status, "existing alternate is reused");
            Eq(first.Path, second.Path, "same alternate");
            True(!File.Exists(primary), "primary not created as a duplicate");
        }

        static void ShortcutCli()
        {
            string box = NewDir("sc-cli");
            string root = FakeRoot(box, "CLI 根");
            string desk = NewDir("sc-cli\\desk");
            StringWriter log = new StringWriter();

            LauncherOptions ok = LauncherOptions.Parse(new string[] { "--install-shortcut", "--shortcut-dir", desk });
            Eq(ExitCodes.Ok, Program.RunInstallShortcut(ok, root, log), "create");
            Eq(ExitCodes.Ok, Program.RunInstallShortcut(ok, root, log), "idempotent");
            True(log.ToString().Contains("shortcut created") && log.ToString().Contains("shortcut unchanged"), "output: " + log);
            True(!File.Exists(ShortcutState.StatePath(root)), "custom dir does not mark desktop state");

            LauncherOptions missing = LauncherOptions.Parse(new string[] { "--install-shortcut", "--shortcut-dir", Path.Combine(box, "不存在") });
            Eq(ExitCodes.ShortcutFailed, Program.RunInstallShortcut(missing, root, null), "missing dir fails, not created");
            True(!Directory.Exists(Path.Combine(box, "不存在")), "directory not created");

            string foreignExe = Path.Combine(NewDir("sc-cli\\other"), "x.exe");
            File.WriteAllBytes(foreignExe, new byte[0]);
            string desk2 = NewDir("sc-cli\\desk2");
            ShortcutManager.WriteLink(Path.Combine(desk2, "RAG 工作台.lnk"), foreignExe, "", foreignExe + ",0");
            ShortcutManager.WriteLink(Path.Combine(desk2, "RAG 工作台 (CLI 根).lnk"), foreignExe, "", foreignExe + ",0");
            LauncherOptions conflict = LauncherOptions.Parse(new string[] { "--install-shortcut", "--shortcut-dir", desk2 });
            Eq(ExitCodes.ShortcutConflict, Program.RunInstallShortcut(conflict, root, null), "conflict");
        }

        static void ShortcutStateMarker()
        {
            string box = NewDir("sc-state");
            string root = FakeRoot(box, "根");
            string state = ShortcutState.StatePath(root);
            True(ShortcutState.ShouldAutoEnsure(state, root), "absent -> auto");

            ShortcutResult done = new ShortcutResult();
            done.Status = ShortcutStatus.Created;
            done.Path = Path.Combine(box, "RAG 工作台.lnk");
            ShortcutState.MarkDone(state, root, done);
            True(File.Exists(state), "state written under data\\ui");
            True(!ShortcutState.ShouldAutoEnsure(state, root), "same root -> no auto");

            string moved = FakeRoot(box, "搬走后的根");
            True(ShortcutState.ShouldAutoEnsure(state, moved), "state from another root -> auto again");

            File.WriteAllText(state, "not json");
            True(ShortcutState.ShouldAutoEnsure(state, root), "corrupt state -> auto");
        }

        // ------------------------------------------------------------ processes

        static int EchoArgs(string[] args)
        {
            using (StreamWriter w = new StreamWriter(Console.OpenStandardOutput(), new UTF8Encoding(false)))
            {
                for (int i = 1; i < args.Length; i++)
                    w.WriteLine("ARG[" + Convert.ToBase64String(Encoding.UTF8.GetBytes(args[i])) + "]");
                w.WriteLine("中文输出");
            }
            return 7;
        }

        static int SpawnSleeper()
        {
            ProcessStartInfo psi = new ProcessStartInfo(System.Reflection.Assembly.GetExecutingAssembly().Location, "--sleep 60000");
            psi.UseShellExecute = false;
            psi.CreateNoWindow = true;
            Process child = Process.Start(psi);
            using (StreamWriter w = new StreamWriter(Console.OpenStandardOutput(), new UTF8Encoding(false)))
                w.WriteLine("PID " + child.Id);
            Thread.Sleep(60000);
            return 0;
        }

        static void ProcessArgs()
        {
            List<string> args = new List<string>();
            args.Add("--echo-args");
            args.AddRange(Tricky);
            ProcessRunner r = new ProcessRunner(selfPath, args, runDir, false, 30000);
            ManualResetEvent done = new ManualResetEvent(false);
            ProcessOutcome outcome = null;
            r.Completed += delegate(ProcessOutcome o) { outcome = o; done.Set(); };
            r.Start();
            True(done.WaitOne(30000), "completed");
            True(!outcome.StartFailed, "started: " + outcome.StartError);
            Eq(7, outcome.ExitCode, "exit code");

            List<string> got = new List<string>();
            foreach (string line in outcome.Stdout.Split(new char[] { '\r', '\n' }, StringSplitOptions.RemoveEmptyEntries))
                if (line.StartsWith("ARG[")) got.Add(Encoding.UTF8.GetString(Convert.FromBase64String(line.Substring(4, line.Length - 5))));
            Eq(Tricky.Length, got.Count, "arg count");
            for (int i = 0; i < Tricky.Length; i++) Eq(Tricky[i], got[i], "arg " + i);
            True(outcome.Stdout.Contains("中文输出"), "UTF-8 stdout decoded");

            ProcessRunner missing = new ProcessRunner(Path.Combine(runDir, "no-such-program.exe"), new List<string>(), runDir, false, 5000);
            ManualResetEvent done2 = new ManualResetEvent(false);
            ProcessOutcome o2 = null;
            missing.Completed += delegate(ProcessOutcome o) { o2 = o; done2.Set(); };
            missing.Start();
            True(done2.WaitOne(10000), "missing program completes");
            True(o2.StartFailed, "missing program reported as start failure");
        }

        static void ProcessCancelTree()
        {
            // 与被测进程无关的“旁观者”进程：取消后必须仍然存活。
            Process bystander = Process.Start(new ProcessStartInfo(selfPath, "--sleep 20000") { UseShellExecute = false, CreateNoWindow = true });
            try
            {
                ProcessRunner r = new ProcessRunner(selfPath, new List<string> { "--spawn-sleeper" }, runDir, true, 0);
                ManualResetEvent sawPid = new ManualResetEvent(false);
                ManualResetEvent done = new ManualResetEvent(false);
                int grandchild = 0;
                ProcessOutcome outcome = null;
                r.Line += delegate(string text, bool isError)
                {
                    if (text.StartsWith("PID ")) { grandchild = int.Parse(text.Substring(4)); sawPid.Set(); }
                };
                r.Completed += delegate(ProcessOutcome o) { outcome = o; done.Set(); };
                r.Start();
                True(sawPid.WaitOne(15000), "grandchild started");

                r.Cancel();
                True(done.WaitOne(15000), "runner completed after cancel");
                True(outcome.Cancelled, "reported as cancelled");
                True(WaitGone(grandchild, 10000), "grandchild terminated with the job");
                True(!bystander.HasExited, "unrelated process untouched");
            }
            finally
            {
                if (!bystander.HasExited) bystander.Kill();   // 仅结束本测试自己启动的旁观者进程
                bystander.Dispose();
            }
        }

        static bool WaitGone(int pid, int timeoutMs)
        {
            Stopwatch sw = Stopwatch.StartNew();
            while (sw.ElapsedMilliseconds < timeoutMs)
            {
                try
                {
                    using (Process p = Process.GetProcessById(pid))
                    {
                        if (p.HasExited) return true;
                    }
                }
                catch (ArgumentException)
                {
                    return true;
                }
                Thread.Sleep(200);
            }
            return false;
        }
    }
}
