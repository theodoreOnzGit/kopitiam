# AID-0008: Visual Basic — which dialects, and a native parser instead of an LSP

* **Status:** Pending review
* **Bead:** `kopitiam-7ef`
* **Date:** 2026-07-14
* **Decided by:** AI (Claude), maintainer absent

Visual Basic was one of four languages added to `kopitiam-semantic` in parallel
(Python, C#, C++, Visual Basic). The other three each drive a real language
server through the existing `crate::lsp_client`, exactly as `rust_analyzer.rs`
does. Visual Basic does not, and the reason is not a technical shortcut. It is
that **no obtainable Visual Basic language server exists.**

That claim is the whole decision, so it is evidenced first.

## The brief's premise, checked

The brief asked me to find out whether a VB.NET server is reachable via
`Microsoft.CodeAnalysis.LanguageServer` "in practice", and warned me not to
fabricate a server that isn't there. Here is what is actually there.

**VB.NET — the Roslyn *compiler* supports it; the Roslyn *language server* does
not.**

This is the trap, and it is a good one, because both halves are true.
`Microsoft.CodeAnalysis.VisualBasic` is a current, shipping NuGet package: the
VB compiler and its full semantic-analysis API are alive and maintained. From
that, everyone reasonably infers that the Roslyn LSP speaks VB.

It does not. `src/LanguageServer/Microsoft.CodeAnalysis.LanguageServer/Microsoft.CodeAnalysis.LanguageServer.csproj`
— the server behind the VS Code C# extension, and the only modern standalone
Roslyn LSP — has these language `ProjectReference`s:

```
Microsoft.CodeAnalysis.CSharp.Features.csproj
```

and that is the complete list. No `Microsoft.CodeAnalysis.VisualBasic.Features`,
no `.VisualBasic.Workspaces`. With no VB language service in its MEF
composition, `LanguageServerProjectSystem` cannot even resolve a language for a
`.vbproj` (it defers to `ProjectFileExtensionRegistry.TryGetLanguageNameFromProjectPath`,
which is populated from the registered language services), so a VB project
cannot be *loaded*, let alone queried.

I did not run the server against a `.vbproj` to confirm the failure — I read the
build graph. That is weaker than an execution trace and I am flagging it as
such, but it is not a close call: a language service that is not linked in
cannot serve requests.

Corroborating, from the humans who own it: **`dotnet/vscode-csharp#25`, "Fully
Support Visual Basic", opened January 2016, closed *Resolved-By Design*.** VB's
IDE experience is delivered in-process inside Visual Studio on Windows. It has
never been exposed over LSP and Microsoft has said, in as many words, that it is
not going to be.

**OmniSharp is a false lead, and a seductive one.** `nvim-lspconfig` lists `vb`
among the `omnisharp` filetypes, which looks like a green light — it is where I
expected to end up. But OmniSharp-Roslyn's codebase is 99.7% C#, its README
offers "C# language services", and the VB feature request
(`OmniSharp/omnisharp-roslyn#1111`, February 2018) was never implemented. The
filetype registration is aspirational. OmniSharp is also being superseded by the
Roslyn LSP, so it is the wrong horse regardless.

**VBA has one real project, which you cannot use as a server.**
`SSlinky/VBA-LanguageServer` is genuine work — ANTLR grammar, TypeScript server,
actively released (v1.7.4, June 2025). But it ships only as a `.vsix` VS Code
extension: there is no binary to put on `PATH`, and driving it would mean
depending on Node and unpacking a VS Code extension at runtime. Rubberduck's
much-discussed LSP is an in-process VBE COM add-in — Windows-only, and not a
server you can spawn from Rust.

**VB6 / classic VB has nothing.** It never did. twinBASIC is a closed IDE, not a
language server.

So the choice was never "LSP or parser". It was **"parser, or no VB support".**

## What was decided

**1. Write a native Rust parser, in `providers/vbnet.rs`. No new dependencies.**

The alternatives, and why each loses:

* *Drive OmniSharp anyway.* Adds a mandatory .NET runtime for a server that
  does not implement the feature. Fails for VBA and VB6 entirely — which is most
  of the code KOPITIAM would actually be asked to translate.
* *Unpack and drive the VBA `.vsix` over Node.* Adds Node, a runtime unpack
  step, and a network fetch, to support one of the three dialects. This is the
  Mason mistake from AID-0003 (a supply chain masquerading as a feature) and it
  breaks Offline First outright.
* *Wait for Microsoft.* Issue #25 is ten years old and closed. This is not a
  waiting game.
* *Skip VB.* CLAUDE.md's crate table names Visual Basic explicitly as a language
  adapter that lives in `kopitiam-semantic`. Skipping it is not on the table.

And the parser is not a consolation prize — it is the *better* answer here, for
a reason worth stating plainly. CLAUDE.md's Translation Platform exists to make
legacy code legible. A dead VB6 codebase is the canonical instance of that
problem, and it is *precisely* the case an LSP was never going to serve, because
nobody is going to write a language server for a language Microsoft killed in
2008. Every other provider borrows its understanding from someone else's tool.
For VB, the runtime has to own it. Which is, word for word, the Semantic Runtime
mission: *the runtime owns understanding.*

VB's grammar makes this tractable rather than heroic: line-oriented,
`Module`/`Class`/`Sub`/`Function`/`End X`, no template metaprogramming, no
preprocessor worth the name. Nothing like C++.

**2. Support VB.NET, VBA and VB6. Do not claim VBScript.**

| Dialect | Extensions | Decision |
|---|---|---|
| VB.NET | `.vb`, `.vbproj` | Supported |
| VBA (Office macros) | `.bas`, `.cls`, `.frm` | Supported |
| VB6 / classic VB | `.bas`, `.cls`, `.frm`, `.vbp` | Supported |
| VBScript | `.vbs` | **Not** supported |

VBA and VB6 share a declaration grammar (VBA *is* VB6's engine embedded in
Office) and are handled by one code path, distinguished from VB.NET only where
they genuinely disagree. VBScript is a related but distinct dialect; parsing it
by accident and claiming it works would be a lie of the kind this project's
Communication rules forbid.

**3. It is a declaration extractor. Statement bodies are skipped, and this is
stated in the rustdoc rather than implied away.**

It parses the declaration surface — types, members, fields, imports,
inheritance, VB6 `Declare` P/Invoke signatures — and skips the insides of
methods. So there is no call graph and no local-variable analysis. That is a
respectable v1 for symbol extraction and the module docs say exactly that, along
with a list of what is *not* covered (`#If` branches are both parsed; `:`
statement separators are not split; implicit VB.NET line continuation is not
handled; `.vbproj` contents are not read).

**4. Inheritance is emitted as `RelationshipKind::Custom("inherits")`.**

The ontology has no `Inherits`/`Extends` variant and `kopitiam-ontology` was off
limits to me. `Implements` maps cleanly onto `ImplementedBy` (interface ->
implementor, matching the direction the ontology's own rustdoc documents). But
`Inherits` does not: flattening "is a" into `DependsOn`'s "uses" loses a
distinction a translation genuinely needs. `Custom` is an existing variant and
carries the fact losslessly.

**This is the one thing I would ask you to change in the ontology**, and it needs
a cross-language decision, not a VB one — C#, C++ and Python all have
inheritance, and if the four adapters each pick a different encoding, the shared
vocabulary has failed at exactly the thing it exists for.

## What would make this wrong

* **If a usable VB.NET language server appears** — or if someone demonstrates
  the Roslyn LSP loading a `.vbproj`, which would mean I read the build graph
  wrong — then for `.vb` files specifically the LSP is the better source of
  truth (real type resolution, real symbol binding, no parser to maintain), and
  this provider should become the fallback rather than the primary. Note this
  would *not* rescue VBA or VB6, so the parser stays either way. The cheap way
  to falsify me: point `Microsoft.CodeAnalysis.LanguageServer` at a `.vbproj`
  and issue a `workspace/symbol`.
* **If you only ever cared about VB.NET** and never about VBA/VB6, then the
  value of a hand-written parser drops a lot, and taking a .NET runtime
  dependency to get *something* out of OmniSharp becomes more defensible than I
  judged. I weighted legacy VB6 heavily because that is where the Translation
  Platform's stated purpose points, but that is my reading of the mission, not
  your instruction.
* **If skipping statement bodies turns out to be the wrong cut** — because what
  you actually want from VB is a call graph for translation planning, not a
  symbol index — then the parser needs a statement layer, and that is a
  materially bigger piece of work that should be scoped deliberately rather than
  bolted on.
* **If `Custom("inherits")` is not how the other three adapters encode
  inheritance**, then the shared ontology is inconsistent and one of us is
  wrong. That is a decision only you can make across all four.
