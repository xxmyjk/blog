---
title: css中清除浮动的几种方式
date: 2016-10-26 18:22:15
tags: [前端, css, 浮动]
---

前端使用 `ul > li` + `float` 方式生成一个 `navbar` 是一种常见的页面展示手段, 但是浮动之后会导致`ul`高度无法正常撑起, 所以需要清除浮动以正常撑起父元素高度. 这里介绍几种常见的浮动清除的方式.

```html
    <ul>
        <li></li>
        <li></li>
        <li></li>
        <li></li>
    </ul>
```

```css
    ul {
        margin: 0 0;
        padding: 0 0;
        list-style-type: none;
    }

    li {
        float: right;
        width: 80px;
        height: 40px;
        margin-right: 5%;
        margin-bottom: 10px;
        line-height: 40px;
        text-align: center;
    }
```

<!-- more -->

### 给ul添加高度

    这个是最直接的方法, 给`ul`元素添加一个高度

    ```css
        ul {
            height: 40px;
        }
    ```

### 给最后一个li后添加一个 **空的** `div`, 给`div`添加`clear: both`样式

    ```html
        <li>
        </li>
        <div style="clear:both;"></div>
    ```

### 给ul添加`overflow: hidden; zoom: 1`样式

    ```css
        ul {
            overflow: hidden;
            zoom: 1;
        }
    ```

### 使用 ul **伪类** 进行浮动清除, 对`ul`添加`class="clearfix"`

    ```css
        .clearfix {
            *zoom: 1;
        }
        .clearfix:before, .clearfix:after {
            display: table;
            line-height: 0;
            content: "";
        }
        .clearfix:after {
            clear: both;
        }
    ```

#### 参考链接
* [推酷](http://www.tuicool.com/articles/3iuaMzn)
