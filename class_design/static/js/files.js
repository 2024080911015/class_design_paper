/**
 * 文件管理 — 基于 jQuery UI
 * 功能：上传、下载、删除
 */
$(function () {
    "use strict";

    var deletePath = "";

    /* ==========================================
       jQuery UI 组件初始化
       ========================================== */

    // 上传 Tabs（第一部分）
    $("#uploadTabs").tabs();

    // 进度条
    $("#uploadProgress").progressbar({ value: false });

    // 消息对话框
    $("#msgDialog").dialog({
        autoOpen: false,
        modal: true,
        width: Math.min(400, $(window).width() - 40),
        buttons: [
            {
                text: "确定",
                class: "primary-btn",
                click: function () {
                    $(this).dialog("close");
                },
            },
        ],
    });

    // 删除确认对话框
    $("#deleteDialog").dialog({
        autoOpen: false,
        modal: true,
        width: Math.min(400, $(window).width() - 40),
        buttons: [
            {
                text: "取消",
                click: function () {
                    $(this).dialog("close");
                },
            },
            {
                text: "删除",
                class: "danger-btn",
                click: function () {
                    $(this).dialog("close");
                    removePath(deletePath);
                },
            },
        ],
    });

    /* ==========================================
       初始化按钮
       ========================================== */
    $("#singleUploadBtn, #folderUploadBtn, #refreshFiles").button();
    $("#refreshFiles").button({ icon: "ui-icon-refresh", showLabel: false });

    /* ==========================================
       Helpers
       ========================================== */
    function showMessage(msg) {
        $("#msgText").text(msg);
        $("#msgDialog").dialog("open");
    }

    function showProgress(show) {
        var $p = $("#uploadProgress");
        if (show) {
            $p.show();
            $p.progressbar("option", "value", false);
        } else {
            $p.hide();
        }
    }

    function encodedPath(path) {
        return path.split("/").map(encodeURIComponent).join("/");
    }

    function formatBytes(bytes) {
        if (!bytes) return "0 B";
        var units = ["B", "KB", "MB", "GB"];
        var i = 0;
        var size = bytes;
        while (size >= 1024 && i < units.length - 1) {
            size /= 1024;
            i++;
        }
        return size.toFixed(i === 0 ? 0 : 1) + " " + units[i];
    }

    function formatTime(ts) {
        var d = new Date(ts * 1000);
        var pad = function (n) { return n < 10 ? "0" + n : n; };
        return d.getFullYear() + "-" + pad(d.getMonth() + 1) + "-" + pad(d.getDate()) + " " + pad(d.getHours()) + ":" + pad(d.getMinutes());
    }

    /* ==========================================
       渲染文件树（文件夹可折叠）
       ========================================== */

    function renderNode(node) {
        var isFolder = node.type === "folder";
        var hasChildren = isFolder && node.children && node.children.length;
        var $row = buildRow(node, isFolder);

        if (!isFolder) return $row;

        var $wrapper = $("<div>").addClass("folder-wrapper");
        $wrapper.append($row);

        if (hasChildren) {
            var $children = $("<div>").addClass("folder-children open");
            node.children.forEach(function (c) {
                $children.append(renderNode(c));
            });
            $wrapper.append($children);
            // 让子元素的按钮也应用 jQuery UI 样式
            $children.find(".dl-btn, .del-btn").button();
        }

        return $wrapper;
    }

    function buildRow(node, isFolder) {
        var $item = $("<div>").addClass("file-item");

        // --- 名称列 ---
        var $name = $("<div>").addClass("name-cell");

        if (isFolder) {
            var $toggle = $("<span>")
                .addClass("collapse-toggle open")
                .html("&#9654;");
            if (!node.children || !node.children.length) {
                $toggle.css("visibility", "hidden");
            }
            $name.append($toggle);
        } else {
            $name.append($("<span>").css({ width: "18px", flexShrink: "0" }));
        }

        var iconHtml = isFolder
            ? '<span class="file-icon folder">F</span>'
            : '<span class="file-icon file">D</span>';
        $name.append(iconHtml);
        $name.append(
            $("<span>").addClass("file-name").attr("title", node.name).text(node.name)
        );
        $item.append($name);

        // --- 大小列 ---
        var sizeText;
        if (isFolder) {
            sizeText = (node.children && node.children.length) ? node.children.length + " 项" : "-";
        } else {
            sizeText = formatBytes(node.size);
        }
        $item.append($("<div>").addClass("size-cell").text(sizeText));

        // --- 日期列 ---
        $item.append($("<div>").addClass("date-cell").text(formatTime(node.modified)));

        // --- 操作列（下载 + 删除）---
        var $actions = $("<div>").addClass("actions-cell");

        var $dlBtn = $("<button>")
            .addClass("dl-btn")
            .text("下载")
            .data("path", node.path);
        $actions.append($dlBtn);

        var $delBtn = $("<button>")
            .addClass("del-btn")
            .text("删除")
            .data("path", node.path)
            .data("name", node.name);
        $actions.append($delBtn);

        $item.append($actions);

        return $item;
    }

    function renderAll(nodes) {
        var $frag = $(document.createDocumentFragment());
        nodes.forEach(function (n) {
            $frag.append(renderNode(n));
        });
        return $frag;
    }

    /* ==========================================
       折叠/展开
       ========================================== */
    function bindCollapse() {
        $("#fileList").on("click", ".collapse-toggle", function (e) {
            e.stopPropagation();
            var $toggle = $(this);
            var $wrapper = $toggle.closest(".folder-wrapper");
            var $children = $wrapper.children(".folder-children");

            if (!$children.length) return;

            if ($children.hasClass("open")) {
                var h = $children[0].scrollHeight;
                $children.css("max-height", h + "px");
                $children[0].offsetHeight;
                $children.removeClass("open");
                $children.css("max-height", "0");
                $toggle.removeClass("open");
            } else {
                $children.addClass("open");
                $children.css("max-height", $children[0].scrollHeight + "px");
                $toggle.addClass("open");
                clearTimeout($children.data("expandTimer"));
                $children.data("expandTimer", setTimeout(function () {
                    $children.css("max-height", "none");
                }, 260));
            }
        });
    }

    /* ==========================================
       加载文件列表
       ========================================== */
    function loadFilesList() {
        $("#fileList").html(
            '<div class="loading-state">加载中...</div>'
        );

        $.get("/files")
            .done(function (files) {
                if (!files.length) {
                    $("#fileList").html('<div class="empty-state">仓库中还没有文件</div>');
                    $("#fileCount").text("");
                    return;
                }
                var $content = renderAll(files);
                $("#fileList").empty().append($content);

                // 对操作按钮应用 jQuery UI button
                $("#fileList .dl-btn").button({ icon: "ui-icon-arrowthickstop-1-s", showLabel: true });
                $("#fileList .del-btn").button({ icon: "ui-icon-trash", showLabel: true });

                var total = countItems(files);
                $("#fileCount").text("共 " + total + " 项");
            })
            .fail(function () {
                $("#fileList").html('<div class="empty-state">加载失败，请重试</div>');
                $("#fileCount").text("");
            });
    }

    function countItems(nodes) {
        var c = 0;
        for (var i = 0; i < nodes.length; i++) {
            c++;
            if (nodes[i].children && nodes[i].children.length) {
                c += countItems(nodes[i].children);
            }
        }
        return c;
    }

    /* ==========================================
       上传功能
       ========================================== */
    function uploadForm($form, url, successMsg) {
        var formData = new FormData($form[0]);
        var $btn = $form.find(".ui-button");
        $btn.button("option", "disabled", true).text("上传中...");
        showProgress(true);

        $.ajax({
            url: url,
            type: "POST",
            data: formData,
            processData: false,
            contentType: false,
        })
            .done(function (resp) {
                showMessage(resp.message || successMsg);
                $form[0].reset();
                $form.closest(".ui-tabs-panel").find(".file-hint").text("尚未选择文件");
                loadFilesList();
            })
            .fail(function (xhr) {
                var resp = xhr.responseJSON || {};
                showMessage(resp.message || "上传失败");
            })
            .always(function () {
                $btn.button("option", "disabled", false).text($btn.data("origText") || "上传");
                showProgress(false);
            });
    }

    /* ==========================================
       删除功能
       ========================================== */
    function removePath(path) {
        if (!path) return;
        $.ajax({
            url: "/delete/" + encodedPath(path),
            type: "POST",
        })
            .done(function (resp) {
                showMessage(resp.message || "删除成功");
                loadFilesList();
            })
            .fail(function (xhr) {
                var resp = xhr.responseJSON || {};
                showMessage(resp.message || "删除失败");
            });
    }

    /* ==========================================
       事件绑定
       ========================================== */

    // --- 文件选择 ---
    $("#singleFileInput").on("change", function () {
        var file = this.files[0];
        $("#singleFileName").text(file ? "已选择: " + file.name : "尚未选择文件");
    });

    $("#folderInput").on("change", function () {
        var count = this.files.length;
        if (!count) { $("#folderFileName").text("尚未选择文件夹"); return; }
        var first = this.files[0].webkitRelativePath || this.files[0].name;
        var name = first.split("/")[0];
        $("#folderFileName").text("已选择: " + name + "（" + count + " 个文件）");
    });

    // --- 上传表单提交 ---
    $("#uploadForm").on("submit", function (e) {
        e.preventDefault();
        uploadForm($(this), "/upload", "文件上传成功");
    });

    $("#uploadFolderForm").on("submit", function (e) {
        e.preventDefault();
        uploadForm($(this), "/upload_folder", "文件夹上传成功");
    });

    // --- 下载（委托，因为按钮是动态渲染的）---
    $("#fileList").on("click", ".dl-btn", function () {
        var path = $(this).data("path");
        if (path) {
            window.location.href = "/download/" + encodedPath(path);
        }
    });

    // --- 删除确认 ---
    $("#fileList").on("click", ".del-btn", function () {
        deletePath = $(this).data("path");
        var name = $(this).data("name");
        $("#deleteTargetText").text("确定要删除「" + name + "」吗？删除后无法恢复。");
        $("#deleteDialog").dialog("open");
    });

    // --- 刷新 ---
    $("#refreshFiles").on("click", loadFilesList);

    /* ==========================================
       启动
       ========================================== */
    bindCollapse();
    loadFilesList();
});
