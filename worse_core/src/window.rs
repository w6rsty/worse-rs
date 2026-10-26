//! Basic window and events
//!
//! # Example
//!
//! ```ignore
//! let window = Arc::new(Window::new(
//!     &WindowDesc::default()
//!         .title("worse")
//!         .initial_size(WIDTH, HEIGHT)
//!         .theme(window::WindowTheme::Dark),
//! )?);
//! ```

use std::{
    cell::{Cell, RefCell},
    collections::VecDeque,
    num::NonZeroIsize,
    rc::Rc,
};

use raw_window_handle::{
    DisplayHandle, HandleError, HasDisplayHandle, HasWindowHandle, RawDisplayHandle, RawWindowHandle,
    Win32WindowHandle, WindowHandle, WindowsDisplayHandle,
};
use thiserror::Error;

#[derive(Debug, Error)]
pub enum Error {
    #[error("failed to register the window class: {0}")]
    ClassRegistration(String),

    #[error("failed to create the window: {0}")]
    Creation(String),
}

#[derive(Debug, Clone, PartialEq)]
pub enum Event {
    // Window
    Resized {
        width: u32,
        height: u32,
    },
    StartResizingMove,
    EndResizingMove,
    Moved {
        x: i32,
        y: i32,
    },
    WindowStateChanged {
        minimized: bool,
        maximized: bool,
    },
    ScaleFactorChanged {
        scale_factor: f32,
    },
    CloseRequested,
    Destroyed,
    Focused(bool),
    RedrawRequested,

    // Keyboard Inputs
    Key {
        virtual_key: u32,
        scancode: u32,
        pressed: bool,
        repeat: bool,
    },
    Char(char), // WM_CHAR

    // Mouse Inputs
    MouseMove {
        x: i32,
        y: i32,
    },
    // MouseRawInput {
    //     delta_x: f32,
    //     delta_y: f32,
    // }, // Raw Input
    // MouseEntered,
    // MouseLeft,
    MouseButton {
        button: MouseButton,
        pressed: bool,
        x: i32,
        y: i32,
    },
    MouseWheel {
        delta: f32,
        x: i32,
        y: i32,
    },
    // File Drag
    // FileDragEnter(PathBuf),
    // FileDragOver(PathBuf),
    // FileDragLeave(PathBuf),
    // FileDrop(PathBuf),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MouseButton {
    Left,
    Right,
    Middle,
    Back,
    Forward,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WindowTheme {
    Light,
    Dark,
}

#[derive(Debug, Clone)]
pub struct WindowDesc {
    pub title: String,
    pub initial_width: u32,
    pub initial_height: u32,
    pub resizable: bool,
    pub visible: bool,
    pub theme: WindowTheme,
}

impl Default for WindowDesc {
    fn default() -> Self {
        Self {
            title: String::from("window"),
            initial_width: 1280,
            initial_height: 720,
            resizable: true,
            visible: true,
            theme: WindowTheme::Dark,
        }
    }
}

impl WindowDesc {
    pub fn title(mut self, title: &str) -> Self {
        self.title = title.to_string();
        self
    }

    pub fn initial_size(mut self, width: u32, height: u32) -> Self {
        self.initial_width = width;
        self.initial_height = height;
        self
    }

    pub fn resizable(mut self, resizable: bool) -> Self {
        self.resizable = resizable;
        self
    }

    pub fn visible(mut self, visible: bool) -> Self {
        self.visible = visible;
        self
    }

    pub fn theme(mut self, theme: WindowTheme) -> Self {
        self.theme = theme;
        self
    }
}

#[derive(Debug, Clone, Copy, Default)]
pub struct Display;

impl HasDisplayHandle for Display {
    fn display_handle(&self) -> Result<DisplayHandle<'_>, HandleError> {
        Ok(unsafe { DisplayHandle::borrow_raw(RawDisplayHandle::Windows(WindowsDisplayHandle::new())) })
    }
}

struct WindowState {
    hwnd: Cell<isize>,
    hinstance: isize,
    queue: RefCell<VecDeque<Event>>,
    size: Cell<(u32, u32)>,
    pending_size: Cell<Option<(u32, u32)>>,
    minimized: Cell<bool>,
    maximized: Cell<bool>,
    scale_factor: Cell<f32>,
    closed: Cell<bool>,
    /// Whether `WM_NCCREATE` took ownership of the leaked `Rc`. Only read on the
    /// `CreateWindowExW` failure path, to decide who has to free it.
    attached: Cell<bool>,
}

/// Not `Send`/`Sync`: pump it on the thread that created it.
pub struct Window {
    state: Rc<WindowState>,
}

pub struct EventIter {
    state: Rc<WindowState>,
}

impl Iterator for EventIter {
    type Item = Event;

    fn next(&mut self) -> Option<Event> {
        self.state.queue.borrow_mut().pop_front()
    }
}

impl Window {
    pub fn new(desc: &WindowDesc) -> Result<Self, Error> {
        win32::register_class()?;

        let state = Rc::new(WindowState {
            hwnd: Cell::new(0),
            hinstance: win32::module_handle()?,
            queue: RefCell::new(VecDeque::new()),
            size: Cell::new((desc.initial_width, desc.initial_height)),
            pending_size: Cell::new(None),
            minimized: Cell::new(false),
            maximized: Cell::new(false),
            scale_factor: Cell::new(1.0),
            closed: Cell::new(false),
            attached: Cell::new(false),
        });

        let hwnd = win32::create_window(desc, &state)?;
        state.hwnd.set(hwnd);

        // Only now can the real DPI be known, so re-fit the frame around the
        // requested client size and take the actual size back from the OS.
        win32::fit_client_area(hwnd, desc);
        state.scale_factor.set(win32::scale_factor(hwnd));
        state.size.set(win32::client_size(hwnd));
        state.pending_size.set(None);
        // Sizing messages sent during creation are noise; the caller already
        // knows the size it asked for.
        state.queue.borrow_mut().clear();

        if desc.visible {
            win32::show(hwnd, true);
        }

        Ok(Self { state })
    }

    /// Drains the OS message queue and returns everything that accumulated.
    ///
    /// Non-blocking: if nothing happened, the iterator is empty and the caller
    /// goes straight on to rendering.
    pub fn pump_events(&self) -> EventIter {
        if !self.state.closed.get() {
            win32::pump(&self.state);
        }

        if let Some(size) = self.state.pending_size.take()
            && size != self.state.size.get()
        {
            self.state.size.set(size);
            // To the front, so a renderer sees the resize before the redraw it
            // has to serve at the new size.
            self.state.queue.borrow_mut().push_front(Event::Resized {
                width: size.0,
                height: size.1,
            });
        }

        EventIter {
            state: self.state.clone(),
        }
    }

    pub fn should_close(&self) -> bool {
        self.state.closed.get()
    }

    /// Destroys the window. Typically called in response to
    /// [`Event::CloseRequested`].
    pub fn close(&self) {
        if let Some(hwnd) = self.hwnd() {
            win32::destroy(hwnd);
        }
    }

    /// Client-area size in physical pixels.
    pub fn size(&self) -> (u32, u32) {
        self.state.size.get()
    }

    pub fn width(&self) -> u32 {
        self.state.size.get().0
    }

    pub fn height(&self) -> u32 {
        self.state.size.get().1
    }

    /// Physical pixels per logical pixel: `1.0` at 96 DPI, `1.5` at 150% scaling.
    pub fn scale_factor(&self) -> f32 {
        self.state.scale_factor.get()
    }

    pub fn set_title(&self, title: &str) {
        if let Some(hwnd) = self.hwnd() {
            win32::set_title(hwnd, title);
        }
    }

    pub fn show(&self) {
        if let Some(hwnd) = self.hwnd() {
            win32::show(hwnd, true);
        }
    }

    pub fn hide(&self) {
        if let Some(hwnd) = self.hwnd() {
            win32::show(hwnd, false);
        }
    }

    /// Asks the OS to deliver an [`Event::RedrawRequested`].
    pub fn request_redraw(&self) {
        if let Some(hwnd) = self.hwnd() {
            win32::invalidate(hwnd);
        }
    }

    fn hwnd(&self) -> Option<isize> {
        match self.state.hwnd.get() {
            0 => None,
            hwnd => Some(hwnd),
        }
    }
}

impl Drop for Window {
    fn drop(&mut self) {
        self.close();
    }
}

impl HasWindowHandle for Window {
    fn window_handle(&self) -> Result<WindowHandle<'_>, HandleError> {
        let hwnd = NonZeroIsize::new(self.state.hwnd.get()).ok_or(HandleError::Unavailable)?;
        let mut handle = Win32WindowHandle::new(hwnd);
        handle.hinstance = NonZeroIsize::new(self.state.hinstance);
        // SAFETY: the handle is live for as long as `&self` is — `close()` and
        // `WM_NCDESTROY` both zero `state.hwnd`, and the check above turns that
        // into `HandleError::Unavailable` rather than a dangling `HWND`.
        Ok(unsafe { WindowHandle::borrow_raw(RawWindowHandle::Win32(handle)) })
    }
}

impl HasDisplayHandle for Window {
    fn display_handle(&self) -> Result<DisplayHandle<'_>, HandleError> {
        Display.display_handle()
    }
}

/// The Win32 backend. Nothing below this line escapes into the public API.
mod win32 {
    use crate::window::WindowTheme;

    use super::{Error, Event, MouseButton, WindowDesc, WindowState};

    use std::{ffi::c_void, ptr::null_mut, rc::Rc, sync::OnceLock};

    use windows::{
        Win32::{
            Foundation::{ERROR_CLASS_ALREADY_EXISTS, GetLastError, HINSTANCE, HWND, LPARAM, LRESULT, RECT, WPARAM},
            Graphics::{
                Dwm::{
                    DWM_SYSTEMBACKDROP_TYPE, DWMSBT_TRANSIENTWINDOW, DWMWA_SYSTEMBACKDROP_TYPE,
                    DWMWA_USE_IMMERSIVE_DARK_MODE, DwmSetWindowAttribute,
                },
                Gdi::{BLACKNESS, BeginPaint, EndPaint, HBRUSH, InvalidateRect, PAINTSTRUCT, PatBlt},
            },
            System::LibraryLoader::GetModuleHandleW,
            UI::{
                HiDpi::{
                    AdjustWindowRectExForDpi, DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2, GetDpiForSystem,
                    GetDpiForWindow, SetProcessDpiAwarenessContext,
                },
                WindowsAndMessaging::{
                    CREATESTRUCTW, CS_HREDRAW, CS_OWNDC, CS_VREDRAW, CW_USEDEFAULT, CreateWindowExW, DefWindowProcW,
                    DestroyWindow, DispatchMessageW, GWLP_USERDATA, GetClientRect, GetWindowLongPtrW, IDC_ARROW,
                    LoadCursorW, MSG, PM_REMOVE, PeekMessageW, RegisterClassExW, SIZE_MAXIMIZED, SIZE_MINIMIZED,
                    SW_HIDE, SW_SHOW, SWP_NOACTIVATE, SWP_NOMOVE, SWP_NOZORDER, SetWindowLongPtrW, SetWindowPos,
                    SetWindowTextW, ShowWindow, TranslateMessage, WHEEL_DELTA, WINDOW_EX_STYLE, WINDOW_STYLE, WM_CHAR,
                    WM_CLOSE, WM_DPICHANGED, WM_ENTERSIZEMOVE, WM_EXITSIZEMOVE, WM_KEYDOWN, WM_KEYUP, WM_KILLFOCUS,
                    WM_LBUTTONDOWN, WM_LBUTTONUP, WM_MBUTTONDOWN, WM_MBUTTONUP, WM_MOUSEMOVE, WM_MOUSEWHEEL, WM_MOVE,
                    WM_NCCREATE, WM_NCDESTROY, WM_PAINT, WM_QUIT, WM_RBUTTONDOWN, WM_RBUTTONUP, WM_SETFOCUS, WM_SIZE,
                    WM_SYSKEYDOWN, WM_SYSKEYUP, WM_XBUTTONDOWN, WM_XBUTTONUP, WNDCLASSEXW, WS_CAPTION, WS_MINIMIZEBOX,
                    WS_OVERLAPPED, WS_OVERLAPPEDWINDOW, WS_SYSMENU, XBUTTON1, XBUTTON2,
                },
            },
        },
        core::{BOOL, HSTRING, PCWSTR, w},
    };

    const CLASS_NAME: PCWSTR = w!("worse_window_class");

    /// Registers the window class exactly once per process, and opts the process
    /// into per-monitor DPI awareness while it is at it.
    ///
    /// The result is cached rather than retried: if the class could not be
    /// registered the first time, it will not register the second time either.
    pub fn register_class() -> Result<(), Error> {
        static CLASS: OnceLock<Result<(), String>> = OnceLock::new();

        CLASS
            .get_or_init(|| unsafe {
                // May legitimately fail if a manifest already set the awareness
                // level, which is fine — we only need it to not be "unaware".
                let _ = SetProcessDpiAwarenessContext(DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2);

                let instance: HINSTANCE = GetModuleHandleW(None).map_err(|e| e.to_string())?.into();
                let cursor = LoadCursorW(None, IDC_ARROW).map_err(|e| e.to_string())?;

                let class = WNDCLASSEXW {
                    cbSize: std::mem::size_of::<WNDCLASSEXW>() as u32,
                    hCursor: cursor,
                    hInstance: instance,
                    lpszClassName: CLASS_NAME,
                    // CS_OWNDC because the GPU wants a device context that is
                    // not recycled out from under the swapchain.
                    style: CS_HREDRAW | CS_VREDRAW | CS_OWNDC,
                    lpfnWndProc: Some(wndproc),
                    hbrBackground: HBRUSH(null_mut()),
                    ..Default::default()
                };

                if RegisterClassExW(&class) == 0 {
                    let code = GetLastError();
                    if code != ERROR_CLASS_ALREADY_EXISTS {
                        return Err(format!("RegisterClassW failed with code {}", code.0));
                    }
                }
                Ok(())
            })
            .clone()
            .map_err(Error::ClassRegistration)
    }

    pub fn module_handle() -> Result<isize, Error> {
        unsafe {
            let instance: HINSTANCE = GetModuleHandleW(None)
                .map_err(|e| Error::Creation(e.to_string()))?
                .into();
            Ok(instance.0 as isize)
        }
    }

    fn window_style(desc: &WindowDesc) -> WINDOW_STYLE {
        if desc.resizable {
            WS_OVERLAPPEDWINDOW
        } else {
            WS_OVERLAPPED | WS_CAPTION | WS_SYSMENU | WS_MINIMIZEBOX
        }
    }

    /// Grows a client-area size into the full window size for a given DPI.
    fn frame_size(width: u32, height: u32, style: WINDOW_STYLE, dpi: u32) -> (i32, i32) {
        let mut rect = RECT {
            left: 0,
            top: 0,
            right: width as i32,
            bottom: height as i32,
        };
        unsafe {
            // On failure `rect` is left untouched, which just means the border
            // eats a few pixels of client area — not worth failing creation for.
            let _ = AdjustWindowRectExForDpi(&mut rect, style, false, WINDOW_EX_STYLE::default(), dpi);
        }
        (rect.right - rect.left, rect.bottom - rect.top)
    }

    pub fn create_window(desc: &WindowDesc, state: &Rc<WindowState>) -> Result<isize, Error> {
        let style = window_style(desc);
        // The window's real DPI is unknowable until it exists and has landed on
        // a monitor, so start from the system DPI and correct afterwards in
        // `fit_client_area`.
        let (width, height) = unsafe { frame_size(desc.initial_width, desc.initial_height, style, GetDpiForSystem()) };

        // Hand one strong count to the window; `WM_NCDESTROY` gives it back.
        let raw = Rc::into_raw(state.clone());

        let result = unsafe {
            CreateWindowExW(
                WINDOW_EX_STYLE::default(),
                CLASS_NAME,
                &HSTRING::from(desc.title.as_str()),
                style,
                CW_USEDEFAULT,
                CW_USEDEFAULT,
                width,
                height,
                None,
                None,
                Some(HINSTANCE(state.hinstance as *mut c_void)),
                Some(raw as *const c_void),
            )
        };

        match result {
            Ok(hwnd) => {
                unsafe {
                    let dark_mode: BOOL = match desc.theme {
                        WindowTheme::Dark => windows::Win32::Foundation::TRUE,
                        WindowTheme::Light => windows::Win32::Foundation::FALSE,
                    };
                    let _ = DwmSetWindowAttribute(
                        hwnd,
                        DWMWA_USE_IMMERSIVE_DARK_MODE,
                        &dark_mode as *const _ as _,
                        std::mem::size_of::<BOOL>() as u32,
                    );

                    let backdrop_type = DWMSBT_TRANSIENTWINDOW;
                    let _ = DwmSetWindowAttribute(
                        hwnd,
                        DWMWA_SYSTEMBACKDROP_TYPE,
                        &backdrop_type as *const _ as _,
                        std::mem::size_of::<DWM_SYSTEMBACKDROP_TYPE>() as u32,
                    );
                };

                Ok(hwnd.0 as isize)
            }
            Err(e) => {
                // If `WM_NCCREATE` never ran, the window never took ownership of
                // the count we leaked above and it is ours to free. If it did
                // run, `WM_NCDESTROY` has already freed it.
                if !state.attached.get() {
                    drop(unsafe { Rc::from_raw(raw) });
                }
                Err(Error::Creation(e.to_string()))
            }
        }
    }

    /// Re-sizes the frame so the client area matches what was asked for at the
    /// window's *actual* DPI.
    pub fn fit_client_area(hwnd: isize, desc: &WindowDesc) {
        let hwnd = hwnd_from(hwnd);
        let dpi = unsafe { GetDpiForWindow(hwnd) };
        let (width, height) = frame_size(desc.initial_width, desc.initial_height, window_style(desc), dpi);
        unsafe {
            let _ = SetWindowPos(
                hwnd,
                None,
                0,
                0,
                width,
                height,
                SWP_NOMOVE | SWP_NOZORDER | SWP_NOACTIVATE,
            );
        }
    }

    pub fn client_size(hwnd: isize) -> (u32, u32) {
        let mut rect = RECT::default();
        unsafe {
            if GetClientRect(hwnd_from(hwnd), &mut rect).is_err() {
                return (0, 0);
            }
        }
        (
            (rect.right - rect.left).max(0) as u32,
            (rect.bottom - rect.top).max(0) as u32,
        )
    }

    pub fn scale_factor(hwnd: isize) -> f32 {
        unsafe { GetDpiForWindow(hwnd_from(hwnd)) as f32 / 96.0 }
    }

    pub fn set_title(hwnd: isize, title: &str) {
        unsafe {
            let _ = SetWindowTextW(hwnd_from(hwnd), &HSTRING::from(title));
        }
    }

    pub fn show(hwnd: isize, visible: bool) {
        unsafe {
            let _ = ShowWindow(hwnd_from(hwnd), if visible { SW_SHOW } else { SW_HIDE });
        }
    }

    pub fn invalidate(hwnd: isize) {
        unsafe {
            let _ = InvalidateRect(Some(hwnd_from(hwnd)), None, false);
        }
    }

    pub fn destroy(hwnd: isize) {
        // Sends `WM_DESTROY`/`WM_NCDESTROY` synchronously, so by the time this
        // returns the state has already been marked closed.
        unsafe {
            let _ = DestroyWindow(hwnd_from(hwnd));
        }
    }

    /// Drains this thread's message queue, dispatching each message to
    /// [`wndproc`], which is what actually fills `state.queue`.
    pub fn pump(state: &Rc<WindowState>) {
        let mut msg = MSG::default();
        unsafe {
            // `None` for the window filter: a thread-wide drain, so messages
            // posted to the thread rather than the window (`WM_QUIT`) are seen.
            while PeekMessageW(&mut msg, None, 0, 0, PM_REMOVE).as_bool() {
                if msg.message == WM_QUIT {
                    state.closed.set(true);
                    break;
                }
                let _ = TranslateMessage(&msg);
                DispatchMessageW(&msg);
            }
        }
    }

    fn hwnd_from(hwnd: isize) -> HWND {
        HWND(hwnd as *mut c_void)
    }

    fn loword(value: usize) -> u16 {
        (value & 0xFFFF) as u16
    }

    fn hiword(value: usize) -> u16 {
        ((value >> 16) & 0xFFFF) as u16
    }

    fn coord(value: u16) -> i32 {
        value as i16 as i32
    }

    unsafe extern "system" fn wndproc(hwnd: HWND, msg: u32, wp: WPARAM, lp: LPARAM) -> LRESULT {
        unsafe {
            if msg == WM_NCCREATE {
                let create = lp.0 as *const CREATESTRUCTW;
                if !create.is_null() {
                    let state = (*create).lpCreateParams as *const WindowState;
                    if !state.is_null() {
                        (*state).attached.set(true);
                        SetWindowLongPtrW(hwnd, GWLP_USERDATA, state as isize);
                    }
                }
                return DefWindowProcW(hwnd, msg, wp, lp);
            }

            let ptr = GetWindowLongPtrW(hwnd, GWLP_USERDATA) as *const WindowState;
            if ptr.is_null() {
                // Messages that arrive before WM_NCCREATE or after WM_NCDESTROY.
                return DefWindowProcW(hwnd, msg, wp, lp);
            }

            // The last message a window receives. Hand the strong count back and
            // make sure nothing can reach the dead `HWND` afterwards.
            if msg == WM_NCDESTROY {
                SetWindowLongPtrW(hwnd, GWLP_USERDATA, 0);
                let state = Rc::from_raw(ptr);
                state.hwnd.set(0);
                state.closed.set(true);
                state.queue.borrow_mut().push_back(Event::Destroyed);
                drop(state);
                return DefWindowProcW(hwnd, msg, wp, lp);
            }

            // SAFETY: borrowed, not reconstructed — the `Rc` count stays with
            // the window until `WM_NCDESTROY` above.
            let state = &*ptr;
            let mut queue = state.queue.borrow_mut();

            match msg {
                WM_SIZE => {
                    let size_type = wp.0 as u32;
                    let minimized = size_type == SIZE_MINIMIZED;
                    let maximized = size_type == SIZE_MAXIMIZED;

                    if minimized != state.minimized.get() || maximized != state.maximized.get() {
                        state.minimized.set(minimized);
                        state.maximized.set(maximized);

                        queue.push_back(Event::WindowStateChanged { minimized, maximized });
                    }

                    // Zero on minimise; there is nothing to render at that size
                    // and reporting it would only make the renderer defend
                    // against it.
                    if !minimized {
                        let (width, height) = (loword(lp.0 as usize) as u32, hiword(lp.0 as usize) as u32);
                        state.pending_size.set(Some((width, height)));
                    }

                    return LRESULT(0);
                }
                WM_ENTERSIZEMOVE => {
                    queue.push_back(Event::StartResizingMove);
                    return LRESULT(0);
                }
                WM_EXITSIZEMOVE => {
                    queue.push_back(Event::EndResizingMove);
                    return LRESULT(0);
                }
                WM_MOVE => {
                    queue.push_back(Event::Moved {
                        x: coord(loword(lp.0 as usize)),
                        y: coord(hiword(lp.0 as usize)),
                    });
                    return LRESULT(0);
                }
                WM_DPICHANGED => {
                    state.scale_factor.set(loword(wp.0) as f32 / 96.0);
                    // lParam is the frame the OS wants us to move to so the
                    // window keeps its physical size across the DPI change.
                    let suggested = lp.0 as *const RECT;
                    if !suggested.is_null() {
                        let rect = &*suggested;
                        let _ = SetWindowPos(
                            hwnd,
                            None,
                            rect.left,
                            rect.top,
                            rect.right - rect.left,
                            rect.bottom - rect.top,
                            SWP_NOZORDER | SWP_NOACTIVATE,
                        );
                    }
                    queue.push_back(Event::ScaleFactorChanged {
                        scale_factor: state.scale_factor.get(),
                    });
                    return LRESULT(0);
                }
                WM_CLOSE => {
                    // Swallow the default (which would destroy the window) and
                    // let the application decide.
                    queue.push_back(Event::CloseRequested);
                    return LRESULT(0);
                }
                WM_SETFOCUS => queue.push_back(Event::Focused(true)),
                WM_KILLFOCUS => queue.push_back(Event::Focused(false)),
                WM_PAINT => {
                    // queue.push_back(Event::RedrawRequested);
                    // // Nothing is painted through GDI — the swapchain owns the
                    // // pixels — but the update region has to be cleared or
                    // // Windows will resend WM_PAINT forever.
                    // let _ = ValidateRect(Some(hwnd), None);

                    let mut ps = PAINTSTRUCT::default();
                    let hdc = BeginPaint(hwnd, &mut ps);

                    let mut rect = RECT::default();
                    GetClientRect(hwnd, &mut rect).unwrap();

                    let _ = PatBlt(hdc, 0, 0, rect.right - rect.left, rect.bottom - rect.top, BLACKNESS);

                    let _ = EndPaint(hwnd, &ps);
                    return LRESULT(0);
                }
                WM_KEYDOWN | WM_SYSKEYDOWN | WM_KEYUP | WM_SYSKEYUP => {
                    let mut scancode = ((lp.0 >> 16) & 0xFF) as u32;
                    let is_extended = ((lp.0 >> 24) & 1) != 0;
                    if is_extended {
                        scancode |= 0xE000;
                    }
                    let pressed = matches!(msg, WM_KEYDOWN | WM_SYSKEYDOWN);
                    let previous_state_down = (lp.0 >> 30) & 1 != 0;
                    let repeat = pressed && previous_state_down;

                    queue.push_back(Event::Key {
                        virtual_key: wp.0 as u32,
                        scancode,
                        pressed,
                        repeat,
                    });
                    // Deliberately falls through to DefWindowProcW: system keys,
                    // Alt+F4 and WM_CHAR synthesis all depend on it.
                }
                WM_CHAR => {
                    let code_unit = wp.0 as u16;
                    if code_unit >= 0x20 && code_unit != 0x7F {
                        if let Some(ch) = char::from_u32(code_unit as u32) {
                            queue.push_back(Event::Char(ch));
                        }
                    }
                    return LRESULT(0);
                }
                WM_MOUSEMOVE => {
                    queue.push_back(Event::MouseMove {
                        x: coord(loword(lp.0 as usize)),
                        y: coord(hiword(lp.0 as usize)),
                    });
                }
                WM_MOUSEWHEEL => {
                    let delta = hiword(wp.0) as i16 as f32 / WHEEL_DELTA as f32;
                    queue.push_back(Event::MouseWheel {
                        delta,
                        x: coord(loword(lp.0 as usize)),
                        y: coord(hiword(lp.0 as usize)),
                    });
                    return LRESULT(0);
                }
                WM_LBUTTONDOWN | WM_LBUTTONUP | WM_RBUTTONDOWN | WM_RBUTTONUP | WM_MBUTTONDOWN | WM_MBUTTONUP
                | WM_XBUTTONDOWN | WM_XBUTTONUP => {
                    let button = match msg {
                        WM_LBUTTONDOWN | WM_LBUTTONUP => MouseButton::Left,
                        WM_RBUTTONDOWN | WM_RBUTTONUP => MouseButton::Right,
                        WM_MBUTTONDOWN | WM_MBUTTONUP => MouseButton::Middle,
                        _ if hiword(wp.0) == XBUTTON1 as u16 => MouseButton::Back,
                        _ if hiword(wp.0) == XBUTTON2 as u16 => MouseButton::Forward,
                        _ => unreachable!(),
                    };
                    let pressed = matches!(msg, WM_LBUTTONDOWN | WM_RBUTTONDOWN | WM_MBUTTONDOWN | WM_XBUTTONDOWN);
                    let (x, y) = (coord(loword(lp.0 as usize)), coord(hiword(lp.0 as usize)));
                    queue.push_back(Event::MouseButton { button, pressed, x, y });
                }
                _ => {}
            }

            // The queue borrow must end before DefWindowProcW, which can
            // re-enter this procedure with another message.
            drop(queue);
            DefWindowProcW(hwnd, msg, wp, lp)
        }
    }
}
