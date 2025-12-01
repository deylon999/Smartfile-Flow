import QtQuick 6.5
import QtQuick.Window 6.5
import QtQuick.Controls 6.5
import QtQuick.Layouts 6.5
import Qt.labs.platform 1.1 as Platform
import Qt5Compat.GraphicalEffects 1.15

Window {
    id: root
    width: 960
    height: 670
    visible: true
    title: "SmartFile Flow"

    // Системный шрифт в зависимости от платформы
    property string appFontFamily: Qt.platform.os === "osx" ? "SF Pro Text"
                                   : Qt.platform.os === "windows" ? "Segoe UI"
                                   : "Inter"

    // Флаг состояния сортировки
    property bool isSorting: false

    // Компонент iOS-переключателя
    Component {
        id: iosSwitchComponent

        Item {
            id: iosSwitch
            width: 50
            height: 28
            property bool checked: false
            signal toggled(bool value)

            Rectangle {
                id: track
                anchors.fill: parent
                radius: height / 2
                color: iosSwitch.checked ? "#0A84FF" : "#E5E5EA"

                Behavior on color { ColorAnimation { duration: 150 } }
            }

            Rectangle {
                id: knob
                width: 24
                height: 24
                radius: 12
                anchors.verticalCenter: parent.verticalCenter
                x: iosSwitch.checked ? parent.width - width - 2 : 2
                color: "#FFFFFF"
                layer.enabled: true
                layer.effect: DropShadow {
                    horizontalOffset: 0
                    verticalOffset: 1
                    radius: 6
                    samples: 16
                    color: "#20000000"
                }

                Behavior on x { NumberAnimation { duration: 150; easing.type: Easing.OutCubic } }
            }

            MouseArea {
                anchors.fill: parent
                cursorShape: Qt.PointingHandCursor
                onClicked: {
                    iosSwitch.checked = !iosSwitch.checked
                    iosSwitch.toggled(iosSwitch.checked)
                }
            }
        }
    }

    color: "#F5F5F7" // фон macOS

    // Основная "карточка" с мягкой тенью
    Rectangle {
        id: card
        anchors.fill: parent
        anchors.margins: 40
        radius: 30
        color: "#FFFFFF" // без бордера — только цвет и тень

        // Тень
        layer.enabled: true
        layer.effect: DropShadow {
            horizontalOffset: 0
            verticalOffset: 2
            radius: 16
            samples: 25
            color: "#20000000"
        }

        ColumnLayout {
            anchors.fill: parent
            anchors.margins: 32
            spacing: 32  // крупные отступы между секциями

            // ====== Заголовок + ML-индикатор ======
            ColumnLayout {
                Layout.fillWidth: true
                spacing: 8

                Text {
                    text: "SmartFile Flow"
                    font.pixelSize: 26
                    font.bold: true
                    font.family: root.appFontFamily
                    color: "#1D1D1F"
                }

                Text {
                    text: "Умная сортировка файлов с ML · минималистичный desktop GUI"
                    font.pixelSize: 13
                    font.bold: false
                    font.family: root.appFontFamily
                    color: "#6E6E73"
                }

                // ML-индикатор как "чип"
                Rectangle {
                    id: mlChip
                    Layout.fillWidth: true
                    Layout.preferredHeight: 32
                    radius: 16
                    // Цвет фона:
                    //  - зелёный, если модель готова (trained)
                    //  - оранжевый, если ML включён, но модель не готова
                    //  - серый, если ML выключен
                    color: appController.mlReady
                           ? "#E8F6EF"
                           : (appController.mlEnabled ? "#FFF5E6" : "#F2F2F7")

                    Behavior on color {
                        ColorAnimation { duration: 200 }
                    }

                    RowLayout {
                        anchors.fill: parent
                        anchors.margins: 10
                        spacing: 8

                        // Точка‑индикатор
                        Rectangle {
                            width: 10
                            height: 10
                            radius: 5
                            // Цвет точки:
                            //  - зелёная, если модель готова
                            //  - оранжевая, если ML включён, но модель не готова
                            //  - серая, если ML выключен
                            color: appController.mlReady
                                   ? "#34C759"
                                   : (appController.mlEnabled ? "#FF9F0A" : "#C7C7CC")
                        }

                        Text {
                            text: appController.mlInfo
                            font.pixelSize: 11
                            font.bold: false
                            font.family: root.appFontFamily
                            color: "#1D1D1F"
                            elide: Text.ElideRight
                            Layout.fillWidth: true
                        }
                    }
                }
            }

            // ====== Блок путей (Откуда / Куда) ======
            ColumnLayout {
                Layout.fillWidth: true
                spacing: 24

                // Откуда
                ColumnLayout {
                    Layout.fillWidth: true
                    spacing: 6

                    Text {
                        text: "Откуда"
                        font.pixelSize: 13
                        font.bold: true
                        font.family: root.appFontFamily
                        color: "#1D1D1F"
                    }

                    RowLayout {
                        Layout.fillWidth: true
                        spacing: 12
                        Layout.alignment: Qt.AlignVCenter

                        // Тонкий macOS‑подобный TextField
                        Rectangle {
                            Layout.fillWidth: true
                            Layout.preferredHeight: 32
                            radius: 8
                            color: "#F5F5F7"
                            border.width: 0

                            TextField {
                                id: sourceField
                                anchors.fill: parent
                                anchors.leftMargin: 10
                                anchors.rightMargin: 10
                                anchors.topMargin: 6
                                anchors.bottomMargin: 6

                                text: appController.sourceDir
                                placeholderText: "Папка с исходными файлами"
                                font.pixelSize: 13
                                font.family: root.appFontFamily
                                color: "#1D1D1F"
                                background: null

                                onEditingFinished: appController.sourceDir = text
                            }
                        }

                        // Кастомная кнопка "Выбрать…"
                        Rectangle {
                            id: sourcePickButton
                            Layout.preferredWidth: 110
                            Layout.preferredHeight: 32
                            radius: 16
                            color: "#F0F4FF"

                            scale: mouseAreaSource.pressed ? 0.95 : 1.0

                            Behavior on color {
                                ColorAnimation { duration: 120 }
                            }
                            Behavior on scale {
                                NumberAnimation { duration: 80; easing.type: Easing.OutCubic }
                            }

                            Text {
                                anchors.centerIn: parent
                                text: "Выбрать…"
                                font.pixelSize: 12
                                font.bold: false
                                font.family: root.appFontFamily
                                color: "#0A84FF"
                            }

                            MouseArea {
                                id: mouseAreaSource
                                anchors.fill: parent
                                hoverEnabled: true
                                cursorShape: Qt.PointingHandCursor

                                onEntered: sourcePickButton.color = "#E1ECFF"
                                onExited: sourcePickButton.color = "#F0F4FF"
                                onClicked: sourceDialog.open()
                            }
                        }
                    }
                }

                // Куда
                ColumnLayout {
                    Layout.fillWidth: true
                    spacing: 6

                    Text {
                        text: "Куда"
                        font.pixelSize: 13
                        font.bold: true
                        font.family: root.appFontFamily
                        color: "#1D1D1F"
                    }

                    RowLayout {
                        Layout.fillWidth: true
                        spacing: 12
                        Layout.alignment: Qt.AlignVCenter

                        Rectangle {
                            Layout.fillWidth: true
                            Layout.preferredHeight: 32
                            radius: 8
                            color: "#F5F5F7"
                            border.width: 0

                            TextField {
                                id: targetField
                                anchors.fill: parent
                                anchors.leftMargin: 10
                                anchors.rightMargin: 10
                                anchors.topMargin: 6
                                anchors.bottomMargin: 6

                                text: appController.targetDir
                                placeholderText: "Папка для отсортированных файлов"
                                font.pixelSize: 13
                                font.family: root.appFontFamily
                                color: "#1D1D1F"
                                background: null

                                onEditingFinished: appController.targetDir = text
                            }
                        }

                        RowLayout {
                            spacing: 12
                            Layout.alignment: Qt.AlignVCenter

                            // Кнопка выбора папки "Куда"
                            Rectangle {
                                id: targetPickButton
                                Layout.preferredWidth: 110
                                Layout.preferredHeight: 32
                                radius: 16
                                color: "#F0F4FF"

                                scale: mouseAreaTarget.pressed ? 0.95 : 1.0

                                Behavior on color {
                                    ColorAnimation { duration: 120 }
                                }
                                Behavior on scale {
                                    NumberAnimation { duration: 80; easing.type: Easing.OutCubic }
                                }

                                Text {
                                    anchors.centerIn: parent
                                    text: "Выбрать…"
                                    font.pixelSize: 12
                                    font.bold: false
                                    font.family: root.appFontFamily
                                    color: "#0A84FF"
                                }

                                MouseArea {
                                    id: mouseAreaTarget
                                    anchors.fill: parent
                                    hoverEnabled: true
                                    cursorShape: Qt.PointingHandCursor

                                    onEntered: targetPickButton.color = "#E1ECFF"
                                    onExited: targetPickButton.color = "#F0F4FF"
                                    onClicked: targetDialog.open()
                                }
                            }

                            // Кнопка "Открыть папку"
                            Rectangle {
                                id: targetOpenButton
                                Layout.preferredWidth: 120
                                Layout.preferredHeight: 32
                                radius: 16
                                color: "#FFFFFF"

                                border.width: 1
                                border.color: "#E0E0E5"

                                scale: mouseAreaOpen.pressed ? 0.95 : 1.0

                                Behavior on color {
                                    ColorAnimation { duration: 120 }
                                }
                                Behavior on scale {
                                    NumberAnimation { duration: 80; easing.type: Easing.OutCubic }
                                }

                                Text {
                                    anchors.centerIn: parent
                                    text: "Открыть папку"
                                    font.pixelSize: 12
                                    font.bold: false
                                    font.family: root.appFontFamily
                                    color: "#1D1D1F"
                                }

                                MouseArea {
                                    id: mouseAreaOpen
                                    anchors.fill: parent
                                    hoverEnabled: true
                                    cursorShape: Qt.PointingHandCursor

                                    onEntered: targetOpenButton.color = "#F2F2F7"
                                    onExited: targetOpenButton.color = "#FFFFFF"
                                    onClicked: appController.openTargetFolder()
                                }
                            }
                        }
                    }
                }
            }

            // Пустое пространство
            Item { Layout.fillHeight: true }

            // ====== Режим работы (копирование / ML / конфликты) ======
            RowLayout {
                Layout.fillWidth: true
                spacing: 32

                // Копирование / перемещение
                ColumnLayout {
                    Layout.fillWidth: true
                    spacing: 6

                    Text {
                        text: "Режим файлов"
                        font.pixelSize: 13
                        font.bold: true
                        font.family: root.appFontFamily
                        color: "#1D1D1F"
                    }

                    RowLayout {
                        spacing: 8
                        Layout.alignment: Qt.AlignVCenter

                        Text {
                            text: appController.copyFiles ? "Копировать файлы" : "Перемещать файлы"
                            font.pixelSize: 12
                            font.family: root.appFontFamily
                            color: "#6E6E73"
                        }

                        Loader {
                            sourceComponent: iosSwitchComponent
                            onLoaded: {
                                item.checked = appController.copyFiles
                                item.toggled.connect(function(val) {
                                    appController.setCopyFiles(val)
                                })
                            }
                        }
                    }
                }

                // ML включен/выключен
                ColumnLayout {
                    Layout.fillWidth: true
                    spacing: 6

                    Text {
                        text: "ML-классификация"
                        font.pixelSize: 13
                        font.bold: true
                        font.family: root.appFontFamily
                        color: "#1D1D1F"
                    }

                    RowLayout {
                        spacing: 8
                        Layout.alignment: Qt.AlignVCenter

                        Text {
                            text: appController.useMl ? "Использовать ML + правила" : "Только правила"
                            font.pixelSize: 12
                            font.family: root.appFontFamily
                            color: "#6E6E73"
                        }

                        Loader {
                            sourceComponent: iosSwitchComponent
                            onLoaded: {
                                item.checked = appController.useMl
                                item.toggled.connect(function(val) {
                                    appController.setUseMl(val)
                                })
                            }
                        }
                    }
                }

                // Стратегия конфликтов
                ColumnLayout {
                    Layout.fillWidth: true
                    spacing: 6

                    Text {
                        text: "Конфликты файлов"
                        font.pixelSize: 13
                        font.bold: true
                        font.family: root.appFontFamily
                        color: "#1D1D1F"
                    }

                    Rectangle {
                        id: conflictButton
                        Layout.fillWidth: true
                        height: 32
                        radius: 16
                        border.width: 1
                        border.color: "#E0E0E5"

                        property string currentLabel: ""
                        property bool hovered: false
                        property bool pressed: false
                        color: hovered ? "#F2F2F7" : "#FFFFFF"
                        scale: pressed ? 0.98 : 1.0

                        Behavior on color {
                            ColorAnimation { duration: 150 }
                        }
                        Behavior on scale {
                            NumberAnimation { duration: 120; easing.type: Easing.OutCubic }
                        }

                        // Центрируем содержимое по вертикали и даём одинаковые отступы слева/справа
                        RowLayout {
                            anchors.verticalCenter: parent.verticalCenter
                            anchors.left: parent.left
                            anchors.right: parent.right
                            anchors.leftMargin: 12
                            anchors.rightMargin: 12
                            spacing: 6

                            Text {
                                text: conflictButton.currentLabel.length > 0 ? conflictButton.currentLabel : "Выбрать стратегию"
                                font.pixelSize: 12
                                font.family: root.appFontFamily
                                color: "#1D1D1F"
                                Layout.fillWidth: true
                                elide: Text.ElideRight
                            }

                            Text {
                                text: "⌄"
                                font.pixelSize: 12
                                color: "#6E6E73"
                            }
                        }

                        MouseArea {
                            anchors.fill: parent
                            cursorShape: Qt.PointingHandCursor
                            hoverEnabled: true
                            onEntered: conflictButton.hovered = true
                            onExited: {
                                conflictButton.hovered = false
                                conflictButton.pressed = false
                            }
                            onPressed: conflictButton.pressed = true
                            onReleased: conflictButton.pressed = false
                            onClicked: conflictMenu.open()
                        }

                        Menu {
                            id: conflictMenu

                            MenuItem {
                                id: renameItem
                                text: "Переименовывать дубликаты"
                                hoverEnabled: true
                                contentItem: Text {
                                    text: renameItem.text
                                    font.pixelSize: 12
                                    font.family: root.appFontFamily
                                    color: "#FFFFFF"
                                    verticalAlignment: Text.AlignVCenter
                                    horizontalAlignment: Text.AlignLeft
                                    anchors.left: parent.left
                                    anchors.leftMargin: 12
                                    anchors.right: parent.right
                                    anchors.rightMargin: 12
                                }
                                background: Rectangle {
                                    implicitHeight: 34
                                    color: renameItem.hovered ? "#3A3A3C" : "transparent"
                                    radius: 6
                                }
                                onTriggered: {
                                    conflictButton.currentLabel = text
                                    appController.setConflictResolution("rename")
                                }
                            }
                            MenuItem {
                                id: skipItem
                                text: "Пропускать дубликаты"
                                hoverEnabled: true
                                contentItem: Text {
                                    text: skipItem.text
                                    font.pixelSize: 12
                                    font.family: root.appFontFamily
                                    color: "#FFFFFF"
                                    verticalAlignment: Text.AlignVCenter
                                    horizontalAlignment: Text.AlignLeft
                                    anchors.left: parent.left
                                    anchors.leftMargin: 12
                                    anchors.right: parent.right
                                    anchors.rightMargin: 12
                                }
                                background: Rectangle {
                                    implicitHeight: 34
                                    color: skipItem.hovered ? "#3A3A3C" : "transparent"
                                    radius: 6
                                }
                                onTriggered: {
                                    conflictButton.currentLabel = text
                                    appController.setConflictResolution("skip")
                                }
                            }
                            MenuItem {
                                id: overwriteItem
                                text: "Перезаписывать файлы"
                                hoverEnabled: true
                                contentItem: Text {
                                    text: overwriteItem.text
                                    font.pixelSize: 12
                                    font.family: root.appFontFamily
                                    color: "#FFFFFF"
                                    verticalAlignment: Text.AlignVCenter
                                    horizontalAlignment: Text.AlignLeft
                                    anchors.left: parent.left
                                    anchors.leftMargin: 12
                                    anchors.right: parent.right
                                    anchors.rightMargin: 12
                                }
                                background: Rectangle {
                                    implicitHeight: 34
                                    color: overwriteItem.hovered ? "#3A3A3C" : "transparent"
                                    radius: 6
                                }
                                onTriggered: {
                                    conflictButton.currentLabel = text
                                    appController.setConflictResolution("overwrite")
                                }
                            }
                        }

                        Component.onCompleted: {
                            switch (appController.conflictResolution) {
                            case "rename":
                                conflictButton.currentLabel = "Переименовывать дубликаты"
                                break
                            case "skip":
                                conflictButton.currentLabel = "Пропускать дубликаты"
                                break
                            case "overwrite":
                                conflictButton.currentLabel = "Перезаписывать файлы"
                                break
                            }
                        }
                    }
                }
            }

            // ====== Статус + прогресс + кнопка ======
            RowLayout {
                Layout.fillWidth: true
                spacing: 16
                Layout.alignment: Qt.AlignVCenter

                ColumnLayout {
                    Layout.fillWidth: true
                    spacing: 8

                    Text {
                        text: root.isSorting ? "Идёт сортировка файлов…" : "Готов к сортировке"
                        font.pixelSize: 12
                        font.bold: false
                        font.family: root.appFontFamily
                        color: root.isSorting ? "#0A84FF" : "#6E6E73"
                    }

                    // Тонкий прогресс-индикатор
                    Rectangle {
                        Layout.fillWidth: true
                        Layout.preferredHeight: 2
                        radius: 1
                        color: "#E5E5EA"

                        Rectangle {
                            id: progressFill
                            anchors.left: parent.left
                            anchors.verticalCenter: parent.verticalCenter
                            height: parent.height
                            width: root.isSorting ? parent.width * 0.45 : parent.width
                            radius: 1
                            color: root.isSorting ? "#0A84FF" : "#34C759"

                            Behavior on width {
                                NumberAnimation {
                                    duration: 260
                                    easing.type: Easing.OutCubic
                                }
                            }
                            Behavior on color {
                                ColorAnimation { duration: 200 }
                            }
                        }
                    }
                }

                // Кнопка сортировки — как iOS
                Rectangle {
                    id: sortButton
                    Layout.preferredWidth: 210
                    Layout.preferredHeight: 44
                    radius: 22

                    property bool down: mouseAreaSort.pressed
                    property bool enabledButton: !root.isSorting

                    // Градиент
                    gradient: Gradient {
                        GradientStop {
                            position: 0.0
                            color: sortButton.enabledButton
                                   ? (sortButton.down ? "#0052A3" : "#0A84FF")
                                   : "#C7C7CC"
                        }
                        GradientStop {
                            position: 1.0
                            color: sortButton.enabledButton
                                   ? (sortButton.down ? "#004080" : "#0066CC")
                                   : "#B0B0B8"
                        }
                    }

                    scale: sortButton.down ? 0.97 : 1.0

                    Behavior on scale {
                        NumberAnimation { duration: 90; easing.type: Easing.OutCubic }
                    }

                    Text {
                        anchors.centerIn: parent
                        text: root.isSorting ? "Сортируем…" : "Сортировать"
                        font.pixelSize: 14
                        font.bold: true
                        font.family: root.appFontFamily
                        color: "white"
                    }

                    MouseArea {
                        id: mouseAreaSort
                        anchors.fill: parent
                        hoverEnabled: true
                        cursorShape: Qt.PointingHandCursor
                        enabled: sortButton.enabledButton

                        onClicked: appController.sortFiles()
                    }

                    opacity: sortButton.enabledButton ? 1.0 : 0.7

                    Behavior on opacity {
                        NumberAnimation { duration: 120 }
                    }
                }
            }

            // ====== Статистика как "чип" ======
            Rectangle {
                id: statsCard
                Layout.fillWidth: true
                Layout.preferredHeight: 40
                radius: 20
                color: "#F5F5F7"

                RowLayout {
                    anchors.fill: parent
                    anchors.margins: 12
                    spacing: 24

                    Text {
                        text: "Статистика"
                        font.pixelSize: 12
                        font.bold: true
                        font.family: root.appFontFamily
                        color: "#1D1D1F"
                    }

                    RowLayout {
                        spacing: 16

                        // 📊 Всего
                        RowLayout {
                            spacing: 4
                            Text {
                                text: "📊"
                                font.pixelSize: 12
                            }
                            Text {
                                text: "Всего:"
                                font.pixelSize: 12
                                font.family: root.appFontFamily
                                color: "#6E6E73"
                            }
                            Text {
                                text: appController.total
                                font.pixelSize: 12
                                font.bold: true
                                font.family: root.appFontFamily
                                color: "#1D1D1F"
                            }
                        }

                        // ✅ Отсортировано
                        RowLayout {
                            spacing: 4
                            Text {
                                text: "✅"
                                font.pixelSize: 12
                            }
                            Text {
                                text: "Отсорт.:"
                                font.pixelSize: 12
                                font.family: root.appFontFamily
                                color: "#6E6E73"
                            }
                            Text {
                                text: appController.sorted
                                font.pixelSize: 12
                                font.bold: true
                                font.family: root.appFontFamily
                                color: "#1D1D1F"
                            }
                        }

                        // ⏭️ Пропущено
                        RowLayout {
                            spacing: 4
                            Text {
                                text: "⏭️"
                                font.pixelSize: 12
                            }
                            Text {
                                text: "Пропущено:"
                                font.pixelSize: 12
                                font.family: root.appFontFamily
                                color: "#6E6E73"
                            }
                            Text {
                                text: appController.skipped
                                font.pixelSize: 12
                                font.bold: true
                                font.family: root.appFontFamily
                                color: "#1D1D1F"
                            }
                        }

                        // ❌ Ошибок
                        RowLayout {
                            spacing: 4
                            Text {
                                text: "❌"
                                font.pixelSize: 12
                            }
                            Text {
                                text: "Ошибок:"
                                font.pixelSize: 12
                                font.family: root.appFontFamily
                                color: "#6E6E73"
                            }
                            Text {
                                text: appController.failed
                                font.pixelSize: 12
                                font.bold: true
                                font.family: root.appFontFamily
                                color: "#1D1D1F"
                            }
                        }
                    }
                }
            }

            // Подключаемся к сигналам контроллера, чтобы обновлять isSorting
            Connections {
                target: appController

                function onSortingStarted() {
                    root.isSorting = true;
                }

                function onSortingFinished() {
                    root.isSorting = false;
                }
            }
        }

        // Диалоги выбора папок (нативные)
        Platform.FolderDialog {
            id: sourceDialog
            title: "Выберите папку с исходными файлами"
            folder: appController.sourceDir

            onAccepted: {
                if (folder) {
                    var path = folder.toLocalFile();
                    appController.sourceDir = path;
                    sourceField.text = path;
                }
            }
        }

        Platform.FolderDialog {
            id: targetDialog
            title: "Выберите папку для отсортированных файлов"
            folder: appController.targetDir

            onAccepted: {
                if (folder) {
                    var path = folder.toLocalFile();
                    appController.targetDir = path;
                    targetField.text = path;
                }
            }
        }
    }
}