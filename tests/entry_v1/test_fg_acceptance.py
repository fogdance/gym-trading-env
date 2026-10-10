from tools.continue_fg_abcd_acceptance import pilot_command, smoke_alive


def test_acceptance_only_launches_pilot_and_missing_smoke_is_finished(tmp_path):
    command = pilot_command(tmp_path / 'manifest.json')
    assert command[command.index('--budget') + 1] == 'pilot'
    assert '--execute' in command and 'formal' not in command
    assert not smoke_alive(0, tmp_path)
