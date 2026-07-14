# AID-0004: devicons on Android — you cannot ship a glyph, you must ship a font

* **Status:** Pending review
* **Bead:** `kopitiam-cj0.8`
* **Date:** 2026-07-14
* **Decided by:** AI (Claude), maintainer absent

## The brief

> For kvim Android compatibility, it does not have pretty devicons by default.
> Such is their terminal. I want the devicons to be shipped in
> kopitiam-neovim, as long as kopitiam-neovim itself is less than 10MB, it is
> fine. So that everything runs with batteries included.

## The technical catch

Devicons are not images and they are not code. They are **codepoints in the
Nerd Fonts Private Use Area** (e.g. the Rust icon is `U+E7A8`). A program
"shipping an icon" only means it prints that codepoint. Whether the user sees
a Rust logo or a tofu box `□` is decided entirely by **the font their terminal
emulator is configured with** — a variable kvim does not control and cannot
write to from inside a running process.

So `nvim-web-devicons` "not working" on Android is not a missing feature in
the plugin. It is a missing font in the terminal. Shipping the *icon table*
into `kopitiam-neovim` (which is trivial — it is a few hundred lines of static
data) would change nothing at all on the maintainer's phone.

Batteries-included therefore has to mean **shipping the font itself**, plus a
way to install it, plus a graceful answer for when it isn't installed.

## What was decided

**A three-tier icon set, chosen automatically.**

| Tier | Requires | Example (Rust file) |
| --- | --- | --- |
| `Nerd` | A Nerd Font in the terminal | `` (U+E7A8) |
| `Unicode` | Any modern font | `◆` / `▪` / `▸` — geometric shapes, colour-coded by filetype |
| `Ascii` | Nothing at all | `[rs]`, `+`, `-` |

Detection is by environment, not by guessing: `TERM`/`TERM_PROGRAM`, an
explicit `KVIM_ICONS=nerd|unicode|ascii` override, and a `NERD_FONT=1`
convention. **When unsure, kvim picks `Unicode`, not `Nerd`.** Guessing wrong
towards Nerd fills the screen with tofu boxes — an unreadable UI — whereas
guessing wrong towards Unicode merely looks plainer. The failure modes are not
symmetric, so the default is the safe one.

**Ship the font, and install it on request.** `kopitiam-neovim` embeds a
patched Nerd Font via `include_bytes!` and gains a `kvim --install-font`
subcommand:

* **On Termux/Android**, it writes `~/.termux/font.ttf` and tells the user to
  run `termux-reload-settings`. This is *the* Android answer: Termux reads
  exactly one font from exactly that path. Note this means the shipped font
  must be a **complete monospace font with the Nerd glyphs patched in**, not a
  symbols-only font — Termux has no font-fallback chain, so a symbols-only
  file would leave them with no letters.
* **On Linux/macOS**, it writes to `~/.local/share/fonts/` (or
  `~/Library/Fonts/`) and runs no font cache command the user didn't ask for.
* It never edits a terminal emulator's config file. Writing to someone's
  `alacritty.toml` unasked is not a battery, it is an intrusion.

**Font choice: JetBrains Mono Nerd Font Mono, Regular only.** ~2 MB, which
sits comfortably inside the 10 MB budget with room for the rest of the crate.
Regular weight only — shipping bold/italic/light would quadruple the size for
a terminal that mostly synthesizes those anyway.

**Licensing.** JetBrains Mono is **OFL-1.1** (SIL Open Font License). This is
compatible with distributing it alongside an AGPLv3 program: the OFL governs
the font as a distinct work, does not infect the program that ships it, and
explicitly permits bundling. The two constraints it *does* impose, both of
which are honoured: the font's copyright and license text must travel with it
(added to `docs/ACKNOWLEDGEMENTS.md` and shipped as a `LICENSE` alongside the
embedded bytes), and the font must not be sold on its own. Nerd Fonts' own
patching tooling is MIT.

## Alternatives considered

* **Ship only the icon table, no font.** This is the literal reading of the
  brief and it accomplishes nothing on Android — the exact platform the
  request was about. Rejected.
* **Symbols-only Nerd Font (~1.2 MB).** Smaller, and correct on desktops where
  the terminal supports a font-fallback chain. Rejected as the *primary*
  answer because Termux does not have a fallback chain, so it would break the
  target platform. It is the better choice on desktop, and can be added later
  as `--install-font --symbols-only` if the size ever matters.
* **Detect Nerd Font support by rendering a glyph and querying the cursor
  position.** This is a real technique (print the glyph, ask the terminal
  where the cursor ended up, infer whether it rendered). It is also fragile,
  racy, flickers on startup, and misbehaves over SSH and inside multiplexers.
  Rejected in favour of the explicit env-var override plus a safe default —
  but noted here because it is clever, and a future maintainer will think of
  it and wonder why it wasn't done.

## What would make this wrong

* If the maintainer's Android terminal is **not Termux** (e.g. Termius, JuiceSSH,
  or a self-hosted terminal app), the `~/.termux/font.ttf` path is wrong and
  the install step needs a different target. I assumed Termux because it is
  overwhelmingly the standard way to run a Rust toolchain on Android, but I did
  not verify it. **This is the most likely thing to need correcting.**
* ~~If 2 MB of embedded font in every build is unwelcome, the font should move
  behind a default-on cargo feature.~~ **ASKED AND ANSWERED (2026-07-14): the
  maintainer wants the font to ship to everyone, unconditionally.** No feature
  gate. Batteries-included means batteries-included, and a desktop user paying
  2.4 MB once is a smaller cost than an Android user discovering their editor
  renders tofu. Do not "optimise" this away later — it is a deliberate choice,
  not an oversight.
