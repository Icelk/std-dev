use std::fs;
use std::io;

use clap::ValueHint;
use clap::{Arg, ArgMatches, Command};
use clap_complete::Shell;

pub fn add_subcommand(command: Command) -> Command {
    command.subcommand(
        Command::new("complete")
            .about("Generate completions for the detected/selected shell and put the completions in appropriate directories.")
            .arg(
                Arg::new("print").short('p').long("print").help(
                    "Print the shell completion to stdout instead of writing to default file.",
                ),
            )
            .arg(
                Arg::new("shell")
                    .takes_value(true)
                    .short('s')
                    .long("shell")
                    .help("Explicitly choose which shell to output.")
                    .value_hint(ValueHint::Other)
            ),
    )
}
/// Ignore if this returns [`None`].
/// Exit application when this returns [`Some`], and signal the user with the error, if any.
#[must_use = "check whether or not to exit"]
pub fn test_subcommand(matches: &ArgMatches, mut command: Command) -> Option<Result<(), String>> {
    matches.subcommand_matches("complete").map(|matches| {
        let shell = {
            let mut name = matches
                .value_of("shell")
                .map(Into::into)
                .ok_or(())
                .or_else(|()| {
                    get_shell::get_shell_name().map_err(|_| {
                        "failed to detect shell, please explicitly supply it".to_owned()
                    })
                })?;
            name.make_ascii_lowercase();
            match name.as_str() {
                "bash" => Shell::Bash,
                "fish" => Shell::Fish,
                "zsh" => Shell::Zsh,
                "pwsh" | "powershell" => Shell::PowerShell,
                "elvish" => Shell::Elvish,
                _ => return Err("unsupported explicit shell".into()),
            }
        };
        let bin_name = command
            .get_bin_name()
            .unwrap_or_else(|| command.get_name())
            .to_owned();

        if matches.is_present("print") || matches!(shell, Shell::Elvish | Shell::PowerShell) {
            clap_complete::generate(shell, &mut command, bin_name, &mut io::stdout());
            Ok(())
        } else {
            let mut buffer = Vec::with_capacity(512);
            clap_complete::generate(shell, &mut command, &bin_name, &mut buffer);
            write_shell(shell, &buffer, &bin_name)
                .map_err(|err| format!("insufficient privileges: {}", err))?;
            Ok(())
        }
    })
}
fn write_shell(shell: Shell, data: &[u8], bin_name: &str) -> Result<(), io::Error> {
    let path = match shell {
        Shell::Fish => {
            let dirs = xdg::BaseDirectories::new()?;
            dirs.place_config_file(format!("fish/completions/{bin_name}.fish"))?
        }
        Shell::Bash => format!("/usr/share/bash-completion/completions/{bin_name}").into(),
        Shell::Zsh => format!("/usr/share/zsh/functions/Completion/Base/_{bin_name}").into(),
        _ => unreachable!("trying to write unsupported shell"),
    };

    println!("Writing completions to {}", path.display());
    fs::write(path, data)?;
    Ok(())
}
