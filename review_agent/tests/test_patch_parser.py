from __future__ import annotations

from review_agent.patch_parser import build_prepass_result, parse_unified_diff


def test_parse_unified_diff_modified_file():
    patch = """diff --git a/repos/repoA/src/a.cpp b/repos/repoA/src/a.cpp
index abc..def 100644
--- a/repos/repoA/src/a.cpp
+++ b/repos/repoA/src/a.cpp
@@ -1,2 +1,3 @@
 int value = 1;
-foo();
+foo();
+bar::baz();
"""
    changes = parse_unified_diff(patch)
    assert len(changes) == 1
    change = changes[0]
    assert change.change_type.value == "modified"
    assert change.old_path == "repos/repoA/src/a.cpp"
    assert change.new_path == "repos/repoA/src/a.cpp"
    assert len(change.hunks) == 1
    assert any(line.kind == "add" and line.new_line == 3 for line in change.hunks[0].lines)


def test_parse_unified_diff_rename_and_delete():
    patch = """diff --git a/old.cpp b/new.cpp
similarity index 100%
rename from old.cpp
rename to new.cpp
diff --git a/dead.cpp b/dead.cpp
deleted file mode 100644
--- a/dead.cpp
+++ /dev/null
@@ -1 +0,0 @@
-int dead();
"""
    changes = parse_unified_diff(patch)
    assert len(changes) == 2
    assert changes[0].change_type.value == "renamed"
    assert changes[1].change_type.value == "deleted"


def test_build_prepass_result_extracts_seeds_and_calls():
    patch = """diff --git a/repos/repoA/src/a.cpp b/repos/repoA/src/a.cpp
--- a/repos/repoA/src/a.cpp
+++ b/repos/repoA/src/a.cpp
@@ -1 +1,4 @@
-foo();
+#include \"x.h\"
+auth::Session::Start();
+doLogin(user);
+static std::atomic<int> g_state{0};
"""
    changes = parse_unified_diff(patch)
    prepass = build_prepass_result(changes, max_symbols=8)
    assert "repos/repoA/src/a.cpp" in prepass.changed_files
    assert any(seed.symbol == "auth::Session::Start" for seed in prepass.seed_symbols)
    assert "doLogin" in prepass.added_call_sites
    assert any(anchor.kind in {"atomic", "concurrency"} for anchor in prepass.suspicious_anchors)


def test_build_prepass_result_filters_std_and_system_noise():
    patch = """diff --git a/src/name_pipe.cpp b/src/name_pipe.cpp
--- a/src/name_pipe.cpp
+++ b/src/name_pipe.cpp
@@ -1 +1,6 @@
+if (initialized_.load()) {
+  LOG_ERROR("bad");
+  auto cb = std::move(handler);
+  SetEvent(h_stop_event_);
+  handleMessage();
+}
"""
    changes = parse_unified_diff(patch)
    prepass = build_prepass_result(changes, max_symbols=16)
    seed_symbols = {seed.symbol for seed in prepass.seed_symbols}
    assert "handleMessage" in seed_symbols
    assert "load" not in seed_symbols
    assert "SetEvent" not in seed_symbols
    assert "std::move" not in seed_symbols
    assert "LOG_ERROR" not in seed_symbols
    assert "handleMessage" in prepass.added_call_sites
    assert "load" not in prepass.added_call_sites
    assert "initialized_.load" in prepass.added_call_sites
    assert any(call.receiver == "initialized_" and call.member == "load" for call in prepass.member_call_sites)
    assert prepass.changed_methods == []


def test_build_prepass_result_keeps_user_defined_load_declaration():
    patch = """diff --git a/src/loader.cpp b/src/loader.cpp
--- a/src/loader.cpp
+++ b/src/loader.cpp
@@ -1 +1,4 @@
+bool load() {
+  return readConfig();
+}
 """
    changes = parse_unified_diff(patch)
    prepass = build_prepass_result(changes, max_symbols=8)
    assert any(decl.symbol == "load" for decl in prepass.changed_declarations)
    assert any(seed.symbol == "load" and seed.relevance_tier == "declaration" for seed in prepass.seed_symbols)


def test_build_prepass_result_distinguishes_declarations_from_member_calls():
    patch = """diff --git a/src/controller.cpp b/src/controller.cpp
--- a/src/controller.cpp
+++ b/src/controller.cpp
@@ -1 +1,5 @@
+class Controller {
+ public:
+  void Run();
+};
+foo.bar();
 """
    changes = parse_unified_diff(patch)
    prepass = build_prepass_result(changes, max_symbols=12)
    assert "Controller::Run" in prepass.changed_methods
    assert "foo.bar" in prepass.added_call_sites
    assert all(decl.symbol != "bar" for decl in prepass.changed_declarations)
