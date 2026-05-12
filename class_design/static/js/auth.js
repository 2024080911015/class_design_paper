$(function () {
    "use strict";

    // 初始化 jQuery UI 按钮
    $("#loginForm button, #registerForm button").button();

    function showError(msg) {
        $("<div>")
            .text(msg)
            .dialog({
                title: "提示",
                modal: true,
                width: Math.min(380, $(window).width() - 40),
                buttons: [
                    {
                        text: "确定",
                        class: "primary-btn",
                        click: function () {
                            $(this).dialog("close").remove();
                        },
                    },
                ],
                close: function () {
                    $(this).remove();
                },
            });
    }

    $("#loginForm, #registerForm").on("submit", function (e) {
        e.preventDefault();

        var $form = $(this);
        var $btn = $form.find("button");
        var origText = $btn.text();

        $btn.button("option", "disabled", true).text("处理中...");

        $.ajax({
            url: $form.attr("action"),
            type: "POST",
            data: $form.serialize(),
        })
            .done(function (resp) {
                if (resp.message === "登录成功") {
                    window.location.href = "/";
                    return;
                }
                if (resp.message === "注册成功") {
                    window.location.href = "/login";
                    return;
                }
                showError(resp.message || "操作成功");
            })
            .fail(function (xhr) {
                var resp = xhr.responseJSON || {};
                showError(resp.message || "操作失败，请检查输入");
            })
            .always(function () {
                $btn.button("option", "disabled", false).text(origText);
            });
    });
});
