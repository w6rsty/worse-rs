use worse_core::log;

fn main() {
    use worse_core::log::*;
    use worse_core::window::*;

    let _log_guard = log::init({
        let mut cfg = log::LogConfig::cli_preset();
        cfg.filter = "debug".to_string();
        cfg
    })
    .unwrap();

    let window = Window::new(
        &WindowDesc::default()
            .title("worse")
            .initial_size(800, 600)
            .resizable(false)
            .theme(WindowTheme::Light),
    )
    .unwrap();

    let names = ["toay", "seos", "iiyno"];
    let mut index = 0;

    let mut exit = false;
    while !exit {
        for event in window.pump_events() {
            info!("{event:?}");

            match event {
                Event::CloseRequested => exit = true,
                Event::Char(char) if char == 'c' => {
                    window.set_title(names[index]);
                    index = (index + 1) % names.len();
                }
                Event::Key { scancode, .. } if scancode == 1 => exit = true,
                _ => {}
            }
        }
    }
    window.close();
}
